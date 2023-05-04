"""
Mostly copy-paste from https://github.com/microsoft/DeBERTa
"""
import torch
from torch import _softmax_backward_data


class XSoftmax(torch.autograd.Function):
    """ Masked Softmax which is optimized for saving memory
    Args:

        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where True indicate that element will be ignored in the softmax caculation.
        dim (int): The dimenssion that will apply softmax.

    Example::
        import torch
        from DeBERTa.deberta import XSoftmax
        # Make a tensor
        x = torch.randn([4,20,100])
        # Create a mask
        mask = (x>0).int()
        y = XSoftmax.apply(x, mask, dim=-1)

    """

    @staticmethod
    def forward(self, input, mask, dim):
        """
        """
        self.dim = dim
        #rmask = ~(mask.bool())
        # where True indicates positions that will be ignored in softmax
        mask = mask.bool()

        output = input.masked_fill(mask, float('-inf'))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(mask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        """

        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None

