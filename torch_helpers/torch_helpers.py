from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable


# debug helpers
def show_params(self: torch.nn.Module):
    for p in self.parameters():
        print(p)


def freeze(self: torch.nn.Module):
    for p in self.parameters():
        p.requires_grad = False


def thaw(self: torch.nn.Module):
    for p in self.parameters():
        p.requires_grad = True


def assert_equal(a, b, message=None):
    """wrapper for numpy all() method"""
    assert (a.numpy() == b.numpy() if hasattr(b, 'numpy') else b).all(), message


def assert_close(a, b, message=None):
    """wrapper for numpy all_close() method"""
    assert (a.numpy() == b.numpy() if hasattr(b, 'numpy') else b).all_close(), message


# helpers
def is_ndarray(d):
    return isinstance(d, np.ndarray)


def is_not_ndarray(d):
    return not isinstance(d, np.ndarray)


def requires_grad(**args):
    raise DeprecationWarning('deprecated, use torch.no_grad() context manager instead.')
    # d = defaultdict(lambda: None, **args)
    # if d['volatile'] or not d['requires_grad']:
    #     return False
    # return True


def tensorify(d, dtype='float', cuda="auto", **kwargs):
    if not hasattr(d, 'shape'):
        d = np.array(d)
    elif torch.is_tensor(d):  # d is tensor or variable
        return d

    if dtype == 'float':
        tensor_type = torch.FloatTensor
    elif dtype == 'int':
        tensor_type = torch.LongTensor
    else:
        raise Exception('tensor type "{}" is not supported'.format(dtype))

    if cuda == "auto" and torch.cuda.is_available():
        return tensor_type(d, **kwargs).cuda()
    else:
        return tensor_type(d, **kwargs)


def varify(d, dtype='float', cuda='auto', requires_grad=True, **kwargs) -> Variable:
    """takes in an array or numpy ndarray, returns variable with requires_grad=True.
    * To use requires_grad=False, use const instead.
    * To use volatile variable, ues volatile instead.
    """
    d = tensorify(d, dtype, cuda=cuda)
    return Variable(d, requires_grad=requires_grad, **kwargs)


def const(d, **kwargs):
    return varify(d, requires_grad=False, **kwargs)


def volatile(d, **kwargs):
    raise DeprecationWarning("""Deprecated: Use the context manager instead: 
        with torch.no_grad():
            y = x * 10 + z""")
    # return varify(d, volatile=True, **kwargs)


def sample_probs(probs, index):
    """
    :param probs: Size(batch_n, feat_n)
    :param index: Size(batch_n, ), Variable(LongTensor)
    :param dim: integer from probs.size inclusive.
    :return: sampled_probability: Size(batch_n)

    NOTE: scatter does not propagate grad to source. Use mask_select instead.

    Is similar to `[var[row,col] for row, col in enumerate(index)]`.
    ```
    import torch
    import torch_extras
    setattr(torch, 'select_item', torch_extras.select_item)
    var = torch.range(0, 9).view(-1, 2)
    index = torch.LongTensor([0, 0, 0, 1, 1])
    torch.select_item(var, index)
    # [0, 2, 4, 7, 9]
    ```
    """

    assert probs.size()[:-1] == index.size(), 'index should be 1 less dimension than probs.'
    mask = one_hot(index, probs.size()[-1])
    mask = cast(mask, 'byte').detach()
    mask.requires_grad = False
    return torch.masked_select(probs, mask).view(*probs.size()[:-1])


ONE_HOT_EXCEPTION = 'Index can not require gradient as per torch.Tensor.scatter\n' + \
                    'requirement. Use `torch_helpers.sample_probs` for masking\n' + \
                    'and random sampling operations.\n' + \
                    '\n' + \
                    'Original pyTorch Exception ==>> \n' + \
                    '    torch.Tensor.scatter:\n' + \
                    '    {e}'


# done: change type to Variable from Tensor
def one_hot(index: Variable, feat_n: int):
    """
    for one_hot masking of categorical distributions, use mask directly instead.
    :param index:
    :param feat_n:
    :return:
    """
    zeros = Variable(torch.FloatTensor(*index.size(), feat_n).zero_())
    ones = Variable(torch.ones(*zeros.size()))
    try:
        return zeros.scatter(-1, index.unsqueeze(-1).expand_as(zeros), ones)
    except AssertionError as e:
        raise Exception(ONE_HOT_EXCEPTION.format(e=e))


TYPES = {
    "byte": "byte",
    torch.ByteTensor: "byte",
    "str": "char",
    torch.CharTensor: "char",
    "double": "double",
    torch.DoubleTensor: "double",
    "float": "float",
    torch.FloatTensor: "float",
    "int": "int",
    torch.IntTensor: "int",
    "long": "long",
    torch.LongTensor: "long",
    "short": "short",
    torch.ShortTensor: "short"
}


def cast(var, dtype):
    """ Cast a Tensor to the given type.
        ```
        import torch
        import torch_extras

        input = torch.FloatTensor(1)
        target_type = type(torch.LongTensor(1))

        assert type(torch.cast(input, target_type)) is torch.LongTensor
        ```
    """

    if dtype in TYPES:
        return getattr(var, TYPES[dtype])()
    else:
        raise ValueError("Not a Tensor type.")
