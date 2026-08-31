import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

from arch import create_transformer
from dataset import TranslationDataset, causal_mask
from configurations import ( project_settings, checkpoint_path, find_latest_checkpoint, tokenizer_path)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def build_tokenizer(data, settings, language):
    path = tokenizer_path(settings, language)

    if path.exists():
        return Tokenizer.from_file(str(path))

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=2
    )

    tokenizer.train_from_iterator(
        (item["translation"][language] for item in data),
        trainer=trainer
    )
    tokenizer.save(str(path))

    return tokenizer


def prepare_data(settings):
    data = load_dataset(
        settings["dataset_name"],
        f"{settings['source_language']}-"
        f"{settings['target_language']}",
        split="train"
    )

    source_tokenizer = build_tokenizer(
        data, settings, settings["source_language"]
    )

    target_tokenizer = build_tokenizer(
        data, settings, settings["target_language"]
    )

    max_length = settings["sequence_length"] - 2

    data = data.filter(
        lambda x: (
            len(source_tokenizer.encode(
                x["translation"][settings["source_language"]]
            ).ids) <= max_length
            and
            len(target_tokenizer.encode(
                x["translation"][settings["target_language"]]
            ).ids) <= max_length
        )
    )

    train_size = int(len(data) * 0.9)
    val_size = len(data) - train_size

    train_raw, val_raw = random_split(
        data,
        [train_size, val_size]
    )

    train_data = TranslationDataset(
        train_raw,
        source_tokenizer,
        target_tokenizer,
        settings["source_language"],
        settings["target_language"],
        settings["sequence_length"]
    )

    val_data = TranslationDataset(
        val_raw,
        source_tokenizer,
        target_tokenizer,
        settings["source_language"],
        settings["target_language"],
        settings["sequence_length"]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=settings["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False
    )

    print(f"Usable translation pairs: {len(data)}")

    return (
        train_loader,
        val_loader,
        source_tokenizer,
        target_tokenizer
    )

def build_model(settings, source_vocab, target_vocab):
    return create_transformer(
        source_vocab,
        target_vocab,
        settings["sequence_length"],
        settings["sequence_length"],
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        num_heads=settings["num_heads"],
        dropout=settings["dropout"],
        intermediate_size=settings["feed_forward_dim"]
    )


def translate(model, source, source_mask, tokenizer, max_length, device):
    model.eval()

    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        memory = model.encode(source, source_mask)
        output = torch.tensor(
            [[sos]], dtype=torch.long, device=device
        )

        for _ in range(max_length - 1):
            mask = causal_mask(output.size(1)).to(device)

            decoded = model.decode(memory, source_mask, output, mask)

            logits = model.project(decoded[:, -1])
            next_token = logits.argmax(-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)

            if next_token.item() == eos:
                break

    return output.squeeze(0)

def validate(model, loader, tokenizer, max_length, device, writer, step, examples=3):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            source = batch["encoder_input"].to(device)
            mask = batch["encoder_mask"].to(device)

            output = translate( model, source, mask, tokenizer, max_length, device)

            prediction = tokenizer.decode( output.cpu().tolist())
            reference = batch["tgt_text"][0]

            predictions.append(prediction)
            references.append(reference)

            if i < examples:
                print("-" * 70)
                print(f"English : {batch['src_text'][0]}")
                print(f"Spanish : {reference}")
                print(f"Model   : {prediction}")

            if i + 1 >= examples:
                break

    if not predictions:
        return

    cer = CharErrorRate()(predictions, references)
    wer = WordErrorRate()(predictions, references)
    bleu = BLEUScore()(
        predictions,
        [[text] for text in references]
    )

    writer.add_scalar("validation/CER", cer.item(), step)
    writer.add_scalar("validation/WER", wer.item(), step)
    writer.add_scalar("validation/BLEU", bleu.item(), step)

def restore(model, optimizer, settings, device):
    resume = settings["resume_from"]

    if resume == "latest":
        path = find_latest_checkpoint(settings)
    elif resume:
        path = checkpoint_path(settings, resume)
    else:
        path = None

    if path is None or not Path(path).exists():
        return 0, 0

    print(f"Loading checkpoint: {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    return state["epoch"] + 1, state["global_step"]

def save(model, optimizer, settings, epoch, step):
    folder = Path(
        f"{settings['dataset_name']}_"
        f"{settings['checkpoint_dir']}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(
        settings,
        f"{epoch:02d}"
    )

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step
    }, path)

def train_epoch(model, loader, optimizer, loss_fn, vocab_size, device, writer, epoch, step):
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch:02d}")

    for batch in progress:
        source = batch["encoder_input"].to(device)
        target = batch["decoder_input"].to(device)
        source_mask = batch["encoder_mask"].to(device)
        target_mask = batch["decoder_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        memory = model.encode(source, source_mask)
        decoded = model.decode(memory, source_mask, target, target_mask)

        logits = model.project(decoded)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")
        writer.add_scalar("training/loss", loss.item(), step)

        step += 1

    return step

def train(settings):
    device = get_device()

    train_loader, val_loader, src_tokenizer, tgt_tokenizer = (
        prepare_data(settings)
    )

    model = build_model(
        settings,
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size()
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        eps=1e-9
    )

    writer = SummaryWriter(settings["experiment_dir"])

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tgt_tokenizer.token_to_id("[PAD]"),
        label_smoothing=0.1
    ).to(device)

    start_epoch, step = restore(model, optimizer, settings, device)

    for epoch in range(start_epoch, settings["epochs"]):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        step = train_epoch(
            model, train_loader,
            optimizer, loss_fn,
            tgt_tokenizer.get_vocab_size(),
            device, writer,
            epoch, step
        )

        validate(
            model, val_loader,
            tgt_tokenizer,
            settings["sequence_length"],
            device, writer, step
        )

        save(
            model, optimizer,
            settings, epoch, step
        )

    writer.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train(project_settings())