# coding: utf-8
"""
Transformer model for translation
"""

import math

# pylint: disable=no-member,arguments-differ
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from nmt.dataset import Vocabulary
from nmt.common import configured, epsilon
from nmt.encoderdecoder import EncoderDecoder

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_size: int,
        dropout: float = 0.1,
        causual: bool = False
    ):
        nn.Module.__init__(self)

        assert embedding_size % num_heads == 0

        self.head_size = embedding_size // num_heads
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.causual = causual

        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.q_linear = nn.Linear(embedding_size, embedding_size, bias=False)
        self.o_linear = nn.Linear(embedding_size, embedding_size, bias=False)

        self.max_dist = 4
        self.p_params = nn.Parameter(torch.zeros(1, 1, 1, self.head_size, 2 * self.max_dist + 1))

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        return f'mha(embedding_size={self.embedding_size},num_heads={self.num_heads}, causual={self.causual})'

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):

        v_len = v.size(-2)
        q_len = q.size(-2)

        k = self.k_linear(k)
        v = self.v_linear(v)
        q = self.q_linear(q)

        k = k.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(-1, v_len, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(-1, q_len, self.num_heads, self.head_size).transpose(1, 2)

        q = q / math.sqrt(self.head_size)

        scores0 = torch.matmul(q, k.transpose(2, 3))
        scores1 = torch.matmul(q.unsqueeze(3), self.p_params).squeeze(3)

        d = (torch.arange(q_len).unsqueeze(1) - torch.arange(v_len).unsqueeze(0)).clamp(-self.max_dist, self.max_dist) + self.max_dist
        d = d.to(q.device).view(1, 1, q_len, v_len).expand_as(scores0)
        scores = scores0 + scores1.gather(-1, d)

        if self.causual:
            causual_mask = torch.tril(torch.ones(q_len, v_len))
            causual_mask = causual_mask.view(1, q_len, v_len).to(scores.device)
            causual_mask = causual_mask != 0
            if mask is None:
                mask = causual_mask
            else:
                mask = mask & causual_mask

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -math.inf)

        attention = self.softmax(scores)
        attention = self.dropout(attention)

        result = torch.matmul(attention, v) \
            .transpose(1, 2).contiguous() \
            .view(-1, q_len, self.embedding_size)

        result = self.o_linear(result)

        return result

def create_positionwise_feedforward(
    embedding_size: int, ffn_size: int, dropout: float
):
    return nn.Sequential(
        nn.Linear(embedding_size, ffn_size), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(ffn_size, embedding_size)
    )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, num_heads: int, embedding_size: int, ffn_size: int, dropout: float
    ):
        nn.Module.__init__(self)
        self.mha = MultiheadAttention(
            num_heads, embedding_size, dropout=dropout, causual=False
        )
        self.ln0 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.ffn = create_positionwise_feedforward(
            embedding_size, ffn_size, dropout
        )
        self.ln1 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        h = self.ln0(x)
        h = self.mha(h, h, h, x_mask)
        x = self.dropout(h) + x
        h = self.ln1(x)
        h = self.ffn(h)
        h = self.dropout(h) + x
        return h


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, num_heads: int, embedding_size: int, ffn_size: int, dropout: float
    ):
        nn.Module.__init__(self)
        self.mha0 = MultiheadAttention(
            num_heads, embedding_size, dropout=dropout, causual=True
        )
        self.ln0 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.mha1 = MultiheadAttention(
            num_heads, embedding_size, dropout=dropout, causual=False
        )
        self.ln1 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.ffn = create_positionwise_feedforward(
            embedding_size, ffn_size, dropout
        )
        self.ln2 = nn.LayerNorm(embedding_size, eps=epsilon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, x_e, y_mask, x_mask):
        h = self.ln0(y)
        h = self.mha0(h, h, h, y_mask)
        y = self.dropout(h) + y
        h = self.ln1(y)
        h = self.mha1(x_e, x_e, h, x_mask)
        y = self.dropout(h) + y
        h = self.ln2(y)
        h = self.ffn(h)
        h = self.dropout(h) + y
        return h


@configured('model.transformer')
class Model(EncoderDecoder):
    def __init__(
        self,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        n_encoders: int = 6,
        n_decoders: int = 6,
        num_heads: int = 8,
        embedding_size: int = 512,
        ffn_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.2
    ):

        EncoderDecoder.__init__(self)

        self.embedding_scale = math.sqrt(embedding_size)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_embed = nn.Embedding(
            len(src_vocab.itos),
            embedding_size,
            padding_idx=src_vocab.pad_index
        )
        if src_vocab == tgt_vocab:
            self.__tgt_embed = None
            self.o_layer = lambda x: torch.matmul(x, self.src_embed.weight.t())
        else:
            self.__tgt_embed = nn.Embedding(
                len(tgt_vocab.itos),
                embedding_size,
                padding_idx=src_vocab.pad_index
            )
            self.o_layer = lambda x: torch.matmul(x, self.__tgt_embed.weight.t())
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_heads, embedding_size, ffn_size, dropout
                ) for _ in range(n_encoders)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    num_heads, embedding_size, ffn_size, dropout
                ) for _ in range(n_decoders)
            ]
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lne = nn.LayerNorm(embedding_size, eps=epsilon)
        self.lnd = nn.LayerNorm(embedding_size, eps=epsilon)

    @staticmethod
    def short_description():
        return "transformer_relpos"

    @staticmethod
    def description():
        return """
        Transformer with relative positional encodings
        ====================================================
        This is the transformer with relative positional encodings instead
        of fixed distance signal.
        """
        
    @property
    def tgt_embed(self):
        if self.__tgt_embed is not None:
            return self.__tgt_embed
        return self.src_embed

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        super().cpu(*args, **kwargs)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def encode(self, x: Tensor, x_mask: Tensor):
        x = self.src_embed(x) * self.embedding_scale
        x = self.emb_dropout(x)
        for encode in self.encoder_layers:
            x = encode(x, x_mask)
        return self.lne(x)

    def __decode(
        self, y: Tensor, x_e: Tensor, y_mask: Tensor, x_mask: Tensor,
        teacher_forcing: bool
    ):
        tgt_embed = self.tgt_embed
        y = tgt_embed(y) * self.embedding_scale
        y = self.emb_dropout(y)
        for decode in self.decoder_layers:
            y = decode(y, x_e, y_mask, x_mask)
        y = self.lnd(y)
        return y

    def decode_one_step(
        self, y: Tensor, x_e: Tensor, y_mask: Tensor, x_mask: Tensor,
        state: Tensor
    ):
        y = self.__decode(y, x_e, y_mask, x_mask, True)
        y = y[:, -1, :]
        y = self.o_layer(y)
        return F.log_softmax(y, dim=-1), None

    def decode(
        self, y: Tensor, x_e: Tensor, y_mask: Tensor, x_mask: Tensor,
        teacher_forcing: bool
    ):
        y = self.__decode(y, x_e, y_mask, x_mask, teacher_forcing)
        y = self.o_layer(y)
        return F.log_softmax(y, dim=-1)
