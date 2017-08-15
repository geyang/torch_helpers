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

```python
import torch
import torch_helpers as h

x = range(0, 3)
t = h.varify(x, 'int')  # setting a `Float` tensor results in RunTimeError

x = np.arange(0.0, 3.0)
t = h.varify(x)

# varify takes torch tensors
x = torch.randn(4, 1)
t = h.varify(x)

t = h.tensorify([[0, 1], [2, 0]])

v = h.varify([[0, 1]])

h.assert_equal(v.data, [[0, 1]])
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
