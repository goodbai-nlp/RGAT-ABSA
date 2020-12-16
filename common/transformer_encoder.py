"""Base class for encoders and generic multi encoders."""

import torch.nn as nn
import torch
from common.sublayer import PositionwiseFeedForward, MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, key_padding_mask=None, structure=None):
        """
    Args:
       input (`FloatTensor`): set of `key_len`
            key vectors `[batch, seq_len, H]`
       mask: binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       key_padding_mask: binary padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
    return:
       res:  [batch, seq_len, H]
    """

        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(
            input_norm,
            input_norm,
            input_norm,
            mask=mask,
            key_padding_mask=key_padding_mask,
            structure=structure,
        )
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def _check_args(self, src, lengths=None):
        _, n_batch = src.size()
        if lengths is not None:
            (n_batch_,) = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, src_key_padding_mask=None, mask=None, structure=None):
        """ See :obj:`EncoderBase.forward()`"""
        """
    Args:
       src (`FloatTensor`): set of vectors `[batch, seq_len, H]`
       mask: binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       src_key_padding_mask: binary key padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
    return:
       out_trans (`FloatTensor`): `[batch, seq_len, H]`

    """
        # self._check_args(src, lengths)

        out = src  # [B, seq_len, H]

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask, src_key_padding_mask, structure=structure)
        out = self.layer_norm(out)  # [B, seq, H]
        return out


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()

    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .unsqueeze(0)
        .expand(batch_size, max_len)
        >= (lengths.unsqueeze(1))
    ).unsqueeze(1)
