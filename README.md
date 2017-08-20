# Torch Helpers, a set of utilities for PyTorch Machine Learning Workflow

## Todo

### Done

- [x] `h.varify`, returns \<torch.Variable\> requires grad.
- [x] `h.const`, returns \<torch.Variable\> with `requires_grad=False`
- [x] `h.volatile`, returns \<torch.Variable\> with `volatile=True`. 
- [x] `h.one_hot`, encodes one_hot vector
- [x] `h.sample_probs`, applies a mask to a categorical distribution, and passes gradient back through the distribution.
- [x] `h.cast`, casts variables and tensors to a different type. For example `int -> float`.
    
## Installation
```bash
pip install torch_helpers
```

## Usage

Here is a snippet from the test:
```python
import torch
import numpy as np
import torch_helpers as h


def test_varify():
    # varify takes integer range objects
    x = range(0, 3)
    v = h.varify(x, 'int')  # setting a `Float` tensor results in RunTimeError

    # takes float arrays
    x = np.arange(0.0, 3.0)
    v = h.varify(x)

    # takes torch tensors
    x = torch.randn(4, 1)
    t = h.varify(x)

    t = h.varify(x, volatile=True)

    t = h.const(x, volatile=True)
    assert t.requires_grad is False and t.volatile is True

    # You can override the requires_grad flag in constants.
    # This is useful when you want to have a constant by default, but
    # would like to switch to var when a requires_grad flag is passed in.
    t = h.const(x, requires_grad=True)
    assert t.requires_grad is True


# h.one_hot gives you one_hot encoding
def test_one_hot():
    acts = h.const([1, 2], dtype='int')
    n = 3
    oh = h.one_hot(acts, n)
    h.assert_equal(oh.data, h.tensorify([[0., 1., 0.], [0., 0., 1.]]),
                   message="one_hot gives incorrect output {}".format(oh))


# For RL tasks, `h.sample_probs` allows you to back-prop through action probability
def test_mask():
    probs = h.varify([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]])
    acts = h.const([1, 2], dtype='int')
    sampled_probs = h.sample_probs(probs, acts)
    sampled_probs.sum().backward()
    dp = probs.grad.data.numpy()
    assert dp[0, 1] is not None and dp[1, 2] is not None, 'key entries of probs grad should be non-zero'
```

## To Develop

```bash
git clone https://github.com/episodeyang/torch_helpers.git
cd torch_helpers
make dev
```

To test, run
```bash
make test
```

This `make dev` command should build the wheel and install it in your current python environment. Take a look at the [./Makefile](./Makefile) for details.

**To publish**, first update the version number, then do:
```bash
make publish
```
