import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from arch import create_transformer
from datasets import TranslationDataset, causal_mask
from configurations import (
    project_settings,
    find_latest_checkpoint,
    tokenizer_path
)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_resources(settings, device):
    source_tokenizer = Tokenizer.from_file(
        str(tokenizer_path(
            settings,
            settings["source_language"]
        ))
    )

    target_tokenizer = Tokenizer.from_file(
        str(tokenizer_path(
            settings,
            settings["target_language"]
        ))
    )

    model = create_transformer(
        source_tokenizer.get_vocab_size(),
        target_tokenizer.get_vocab_size(),
        settings["sequence_length"],
        settings["sequence_length"],
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        num_heads=settings["num_heads"],
        dropout=settings["dropout"],
        intermediate_size=settings["feed_forward_dim"]
    ).to(device)

    checkpoint = find_latest_checkpoint(settings)

    if checkpoint is None:
        raise FileNotFoundError(
            "No trained checkpoint was found."
        )

    state = torch.load(
        checkpoint,
        map_location=device
    )

    model.load_state_dict(
        state["model_state_dict"]
    )

    return model, source_tokenizer, target_tokenizer


def prepare_source(text, tokenizer, sequence_length, device):
    ids = tokenizer.encode(text).ids

    if len(ids) > sequence_length - 2:
        raise ValueError(
            f"Input exceeds the maximum length of "
            f"{sequence_length - 2} tokens."
        )

    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")
    pad = tokenizer.token_to_id("[PAD]")

    tokens = [sos, *ids, eos]
    tokens += [pad] * (sequence_length - len(tokens))

    source = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    mask = (source != pad).unsqueeze(1).unsqueeze(1)

    return source, mask


def generate_translation(
    model,
    source,
    source_mask,
    tokenizer,
    max_length,
    device
):
    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")

    memory = model.encode(
        source,
        source_mask
    )

    generated = torch.tensor(
        [[sos]],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_length - 1):
        target_mask = causal_mask(
            generated.size(1)
        ).to(device)

        decoded = model.decode(
            memory,
            source_mask,
            generated,
            target_mask
        )

        logits = model.project(
            decoded[:, -1]
        )

        next_token = logits.argmax(
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if next_token.item() == eos:
            break

    return generated


def get_example(
    value,
    settings,
    source_tokenizer,
    target_tokenizer
):
    dataset = load_dataset(
        settings["dataset_name"],
        f"{settings['source_language']}-"
        f"{settings['target_language']}",
        split="train"
    )

    dataset = TranslationDataset(
        dataset,
        source_tokenizer,
        target_tokenizer,
        settings["source_language"],
        settings["target_language"],
        settings["sequence_length"]
    )

    example = dataset[int(value)]

    return example["src_text"], example["tgt_text"]


def translate(input_text):
    settings = project_settings()
    device = get_device()

    print(f"Using device: {device}")

    model, source_tokenizer, target_tokenizer = load_resources(
        settings,
        device
    )

    reference = None
    example_id = None

    if isinstance(input_text, int) or input_text.isdigit():
        example_id = int(input_text)

        input_text, reference = get_example(
            example_id,
            settings,
            source_tokenizer,
            target_tokenizer
        )

    source, source_mask = prepare_source(
        input_text,
        source_tokenizer,
        settings["sequence_length"],
        device
    )

    model.eval()

    with torch.no_grad():
        output = generate_translation(
            model,
            source,
            source_mask,
            target_tokenizer,
            settings["sequence_length"],
            device
        )

    prediction = target_tokenizer.decode(
        output.squeeze(0).cpu().tolist()
    )

    print("-" * 70)

    if example_id is not None:
        print(f"Example : {example_id}")

    print(f"English : {input_text}")

    if reference:
        print(f"Spanish : {reference}")

    print(f"Model   : {prediction}")
    print("-" * 70)

    return prediction


if __name__ == "__main__":
    warnings = __import__("warnings")
    warnings.filterwarnings("ignore")

    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I am not a very good student."
    )

    translate(text)