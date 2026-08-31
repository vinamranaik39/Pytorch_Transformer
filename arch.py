import math
import torch
import torch.nn as nn

class FeatureNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias

class DenseNetwork(nn.Module):
    def __init__(self, size, hidden, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, size)
        )

    def forward(self, x):
        return self.layers(x)

class TokenLookup(nn.Module):
    def __init__(self, vocab_size, size):
        super().__init__()
        self.scale = math.sqrt(size)
        self.embedding = nn.Embedding(vocab_size, size)

    def forward(self, x):
        return self.embedding(x) * self.scale

class PositionSignal(nn.Module):
    def __init__(self, size, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, size, 2).float() *
                        (-math.log(10000.0) / size))
        encoding = torch.zeros(max_len, size)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.encoding[:, :x.size(1)])

class AttentionEngine(nn.Module):
    def __init__(self, size, heads, dropout):
        super().__init__()
        if size % heads:
            raise ValueError("size must be divisible by heads")

        self.size, self.heads = size, heads
        self.head_size = size // heads
        self.query = nn.Linear(size, size, bias=False)
        self.key = nn.Linear(size, size, bias=False)
        self.value = nn.Linear(size, size, bias=False)
        self.output = nn.Linear(size, size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_map = None

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        def split(x):
            return x.view(
                batch, -1, self.heads, self.head_size
            ).transpose(1, 2)

        query, key, value = map(split, (query, key, value))

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = self.dropout(torch.softmax(scores, dim=-1))
        self.attention_map = weights

        output = weights @ value
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, -1, self.size)

        return self.output(output)

class SkipNorm(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = FeatureNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))

class EncoderUnit(nn.Module):
    def __init__(self, size, heads, hidden, dropout):
        super().__init__()
        self.attention = AttentionEngine(size, heads, dropout)
        self.feed_forward = DenseNetwork(size, hidden, dropout)

        self.connections = nn.ModuleList([
            SkipNorm(size, dropout),
            SkipNorm(size, dropout)
        ])

    def forward(self, x, mask):
        x = self.connections[0](
            x, lambda y: self.attention(y, y, y, mask)
        )
        return self.connections[1](x, self.feed_forward)

class EncoderStack(nn.Module):
    def __init__(self, size, layers):
        super().__init__()
        self.layers = layers
        self.norm = FeatureNorm(size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderUnit(nn.Module):
    def __init__(self, size, heads, hidden, dropout):
        super().__init__()

        self.self_attention = AttentionEngine(
            size, heads, dropout
        )
        self.cross_attention = AttentionEngine(
            size, heads, dropout
        )
        self.feed_forward = DenseNetwork(
            size, hidden, dropout
        )

        self.connections = nn.ModuleList([
            SkipNorm(size, dropout),
            SkipNorm(size, dropout),
            SkipNorm(size, dropout)
        ])

    def forward(self, x, memory, source_mask, target_mask):
        x = self.connections[0](
            x,
            lambda y: self.self_attention(
                y, y, y, target_mask
            )
        )

        x = self.connections[1](
            x,
            lambda y: self.cross_attention(
                y, memory, memory, source_mask
            )
        )

        return self.connections[2](
            x, self.feed_forward
        )

class DecoderStack(nn.Module):
    def __init__(self, size, layers):
        super().__init__()
        self.layers = layers
        self.norm = FeatureNorm(size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(
                x,
                memory,
                source_mask,
                target_mask
            )

        return self.norm(x)

class VocabularyHead(nn.Module):
    def __init__(self, size, vocab_size):
        super().__init__()
        self.projection = nn.Linear(size, vocab_size)

    def forward(self, x):
        return self.projection(x)

class SequenceTranslator(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        source_embed,
        target_embed,
        source_pos,
        target_pos,
        output_head
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.source_embed = source_embed
        self.target_embed = target_embed

        self.source_pos = source_pos
        self.target_pos = target_pos

        self.output_head = output_head

    def encode(self, source, source_mask):
        source = self.source_pos(
            self.source_embed(source)
        )
        return self.encoder(source, source_mask)

    def decode(
        self,
        memory,
        source_mask,
        target,
        target_mask
    ):
        target = self.target_pos(
            self.target_embed(target)
        )

        return self.decoder(
            target,
            memory,
            source_mask,
            target_mask
        )

    def project(self, x):
        return self.output_head(x)

def create_transformer(
    source_vocab_size,
    target_vocab_size,
    source_sequence_length,
    target_sequence_length,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    intermediate_size=2048
):
    source_embed = TokenLookup(
        source_vocab_size,
        hidden_size
    )

    target_embed = TokenLookup(
        target_vocab_size,
        hidden_size
    )

    source_pos = PositionSignal(
        hidden_size,
        source_sequence_length,
        dropout
    )

    target_pos = PositionSignal(
        hidden_size,
        target_sequence_length,
        dropout
    )

    encoder_layers = nn.ModuleList([
        EncoderUnit(
            hidden_size,
            num_heads,
            intermediate_size,
            dropout
        )
        for _ in range(num_layers)
    ])

    decoder_layers = nn.ModuleList([
        DecoderUnit(
            hidden_size,
            num_heads,
            intermediate_size,
            dropout
        )
        for _ in range(num_layers)
    ])

    encoder = EncoderStack(
        hidden_size,
        encoder_layers
    )

    decoder = DecoderStack(
        hidden_size,
        decoder_layers
    )

    output_head = VocabularyHead(
        hidden_size,
        target_vocab_size
    )

    model = SequenceTranslator(
        encoder,
        decoder,
        source_embed,
        target_embed,
        source_pos,
        target_pos,
        output_head
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model