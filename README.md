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
    x = range(0, 3)
    t = h.varify(x, 'int')  # setting a `Float` tensor results in RunTimeError

    x = np.arange(0.0, 3.0)
    t = h.varify(x)

    x = torch.randn(4, 1)
    t = h.varify(x)


def test_one_hot():
    acts = h.const([1, 2], dtype='int')
    n = 3
    oh = h.one_hot(acts, n)
    h.assert_equal(oh.data, h.tensorify([[0., 1., 0.], [0., 0., 1.]]),
                   message="one_hot gives incorrect output {}".format(oh))


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
