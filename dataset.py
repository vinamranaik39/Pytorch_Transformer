import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data, source_tokenizer, target_tokenizer, source_language="en", target_language="es", sequence_length=128):
        
        super().__init__()
        self.data = data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_length = sequence_length

        self.sos_id = target_tokenizer.token_to_id("[SOS]")
        self.eos_id = target_tokenizer.token_to_id("[EOS]")
        self.pad_id = target_tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        translation = self.data[index]["translation"]

        source_text = translation[self.source_language]
        target_text = translation[self.target_language]

        source_ids = self.source_tokenizer.encode(source_text).ids

        target_ids = self.target_tokenizer.encode(target_text).ids

        source_input = self._prepare_source(source_ids)
        decoder_input = self._prepare_decoder(target_ids)
        target_labels = self._prepare_labels(target_ids)

        return {
            "encoder_input": source_input,
            "decoder_input": decoder_input,
            "encoder_mask": self._source_mask(source_input),
            "decoder_mask": self._target_mask(decoder_input),
            "label": target_labels,
            "src_text": source_text,
            "tgt_text": target_text
        }

    def _prepare_source(self, token_ids):
        available = self.sequence_length - 2

        if len(token_ids) > available:
            raise ValueError(
                f"Source sentence exceeds maximum length of "
                f"{available} tokens."
            )

        tokens = [self.sos_id]
        tokens.extend(token_ids)
        tokens.append(self.eos_id)

        padding = self.sequence_length - len(tokens)
        tokens.extend([self.pad_id] * padding)

        return torch.tensor(tokens, dtype=torch.long)

    def _prepare_decoder(self, token_ids):
        available = self.sequence_length - 1

        if len(token_ids) > available:
            raise ValueError(
                f"Target sentence exceeds maximum length of "
                f"{available} tokens."
            )

        tokens = [self.sos_id]
        tokens.extend(token_ids)

        padding = self.sequence_length - len(tokens)
        tokens.extend([self.pad_id] * padding)

        return torch.tensor(tokens, dtype=torch.long)

    def _prepare_labels(self, token_ids):
        tokens = list(token_ids)
        tokens.append(self.eos_id)

        padding = self.sequence_length - len(tokens)
        tokens.extend([self.pad_id] * padding)

        return torch.tensor(tokens, dtype=torch.long)

    def _source_mask(self, tokens):
        return (tokens != self.pad_id).unsqueeze(0).unsqueeze(0)

    def _target_mask(self, tokens):
        padding_mask = (tokens != self.pad_id).unsqueeze(0)
        attention_mask = causal_mask(self.sequence_length)

        return padding_mask & attention_mask

def causal_mask(size):
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))

    return mask.unsqueeze(0)