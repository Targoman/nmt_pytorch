# coding: utf-8
"""
Base encoder decoder model
"""

from torch import Tensor, nn


class EncoderDecoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def encode(self, x: Tensor, x_mask: Tensor):
        raise NotImplementedError(
            'Abstract EncoderDecoder.encode function called'
        )

    def decode_one_step(
        self, y: Tensor, x_e: Tensor, y_mask: Tensor, x_mask: Tensor,
        state: Tensor
    ):
        raise NotImplementedError(
            'Abstract EncoderDecoder.decode_one_step function called'
        )

    def decode(
        self,
        y: Tensor,
        x_e: Tensor,
        y_mask: Tensor,
        x_mask: Tensor,
        teacher_forcing: bool = True
    ):
        raise NotImplementedError(
            'Abstract EncoderDecoder.decode function called'
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            'Direct calling of translation models is prohibited.'
        )
    
    def get_target_embeddings(self):
        return None