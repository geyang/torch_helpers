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

    t = h.varify(x, volatile=True)

    t = h.const(x, volatile=True)
    assert t.requires_grad is False and t.volatile is True

    # You can override the requires_grad flag in constants.
    # This is useful when you want to have a constant by default, but
    # would like to switch to var when a requires_grad flag is passed in.
    t = h.const(x, requires_grad=True)
    assert t.requires_grad is True

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


def test_mask_operations_correctness():
    sampled_acts = h.const([1, 2], dtype='int')
    # first version
    probs_1 = h.varify([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]])  # pre-sampled probability, we compute it's gradient
    act_oh = h.one_hot(sampled_acts, feat_n=probs_1.size()[-1]).detach()
    act_oh.requires_grad = False
    sampled_probs_1 = probs_1.mul(act_oh).sum(dim=-1).squeeze(dim=-1)
    sampled_probs_1.sum().backward()
    # second version
    probs_2 = h.varify([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1]])
    sampled_probs_2 = h.sample_probs(probs_2, sampled_acts)
    sampled_probs_2.sum().backward()

    assert (sampled_probs_1.data.numpy() == sampled_probs_2.data.numpy()).all(), 'two should give the same result'
    assert (probs_1.grad.data.numpy() ==
            probs_2.grad.data.numpy()).all(), 'two should give the same grad for the original input'
