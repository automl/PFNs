import random
from collections import OrderedDict
import torch

from .. import utils
from ..priors.hebo_prior import Warp
from gpytorch.priors import LogNormalPrior

from botorch.optim import module_to_array, set_params_with_array
import scipy
from scipy.optimize import Bounds
from typing import OrderedDict
import numpy as np
from functools import partial

device = 'cpu:0'


def fit_lbfgs(x, w, nll, num_grad_steps=10, ignore_prior=True, params0=None):
    bounds_ = {}
    if hasattr(w, "named_parameters_and_constraints"):
        for param_name, _, constraint in w.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound
    params0_, property_dict, bounds_ = module_to_array(
        module=w, bounds=bounds_, exclude=None
    )
    if params0 is None: params0 = params0_
    bounds = Bounds(lb=bounds_[0], ub=bounds_[1], keep_feasible=True)

    def loss_f(params, w):
        w = set_params_with_array(w, params, property_dict)
        w.requires_grad_(True)
        loss = 0.
        if not ignore_prior:
            for name, module, prior, closure, _ in w.named_priors():
                prior_term = prior.log_prob(closure(module))
                loss -= prior_term.sum(dim=-1)
        negll = nll(w(x.to(torch.float64)).to(torch.float)).sum()
        #if loss != 0.:
        #    print(loss.item(), negll.item())
        loss = loss + negll
        return w, loss

    def opt_f(params, w):
        w, loss = loss_f(params, w)

        w.zero_grad()
        loss.backward()
        grad = []
        param_dict = OrderedDict(w.named_parameters())

        for p_name in property_dict:
            t = param_dict[p_name].grad
            if t is None:
                # this deals with parameters that do not affect the loss
                grad.append(np.zeros(property_dict[p_name].shape.numel()))
            else:
                grad.append(t.detach().view(-1).cpu().double().clone().numpy())
        w.zero_grad()
        # print(neg_mean_acq.detach().numpy(), x_eval.grad.detach().view(*x.shape).numpy())
        return loss.item(), np.concatenate(grad)

    if num_grad_steps:
        return scipy.optimize.minimize(partial(opt_f, w=w), params0, method='L-BFGS-B', jac=True, bounds=bounds,
                                       options={'maxiter': num_grad_steps})
    else:
        with torch.no_grad():
            return loss_f(params0, w), params0


def log_vs_nonlog(x, w, *args, **kwargs):
    if "true_nll" in kwargs:
        true_nll = kwargs["true_nll"]
        del kwargs["true_nll"]
    else:
        true_nll = None
    params, property_dict, _ = module_to_array(module=w)
    no_log = np.ones_like(params)
    log = np.array([1.9, 0.11] * (int(len(property_dict) / 2)))
    loss_no_log = fit_lbfgs(x, w, *args, **{**kwargs, 'num_grad_steps': 0}, params0=no_log)
    loss_log = fit_lbfgs(x, w, *args, **{**kwargs, 'num_grad_steps': 0}, params0=log)
    print("loss no log", loss_no_log[0][1], "loss log", loss_log[0][1])
    if loss_no_log[0][1] < loss_log[0][1]:
        set_params_with_array(module=w, x=loss_no_log[1], property_dict=property_dict)
    if true_nll:
        best_params, _, _ = module_to_array(module=w)
        print("true nll", fit_lbfgs(x, w, true_nll, **{**kwargs, 'num_grad_steps': 0}, params0=best_params))


def fit_lbfgs_with_restarts(x, w, *args, old_solution=None, rs_size=50, **kwargs):
    if "true_nll" in kwargs:
        true_nll = kwargs["true_nll"]
        del kwargs["true_nll"]
    else:
        true_nll = None
    rs_results = []
    if old_solution:
        rs_results.append(fit_lbfgs(x, old_solution, *args, **{**kwargs, 'num_grad_steps': 0}))
    for i in range(rs_size):
        with torch.no_grad():
            w.concentration0[:] = w.concentration0_prior()
            w.concentration1[:] = w.concentration1_prior()
        rs_results.append(fit_lbfgs(x, w, *args, **{**kwargs, 'num_grad_steps': 0}))
    best_r = min(rs_results, key=lambda r: r[0][1])
    print('best r', best_r)
    with torch.set_grad_enabled(True):
        r = fit_lbfgs(x, w, *args, **kwargs, params0=best_r[1])
    _, property_dict, _ = module_to_array(module=w)
    set_params_with_array(module=w, x=r.x, property_dict=property_dict)
    print('final r', r)
    if true_nll:
        print("true nll", fit_lbfgs(x, w, true_nll, **{**kwargs, 'num_grad_steps': 0}, params0=r.x))
    return r


# use seed 0 for sampling indices, and reset seed afterwards
old_seed = random.getstate()
random.seed(0)
one_out_indices_sampled_per_num_obs = [None]+[random.sample(range(i), min(10, i)) for i in range(1, 100)]
random.setstate(old_seed)

# use seed 0 for sampling subsets
old_seed = random.getstate()
random.seed(0)
subsets = [None]+[[random.sample(range(i), i//2) for _ in range(10)] for i in range(1, 100)]
neg_subsets = [None]+[[list(set(range(i)) - set(s)) for s in ss] for i, ss in enumerate(subsets[1:], 1)]
random.setstate(old_seed)



def fit_input_warping(model, x, y, nll_type='fast', old_solution=None, opt_method="lbfgs", **kwargs):
    """

    :param model:
    :param x: shape (n, d)
    :param y: shape (n, 1)
    :param nll_type:
    :param kwargs: Possible kwargs: `num_grad_steps`, `rs_size`
    :return:
    """
    device = x.device
    assert y.device == device, y.device

    model.requires_grad_(False)

    w = Warp(range(x.shape[1]),
             concentration1_prior=LogNormalPrior(torch.tensor(0.0, device=device), torch.tensor(.75, device=device)),
             concentration0_prior=LogNormalPrior(torch.tensor(0.0, device=device), torch.tensor(.75, device=device)),
             eps=1e-12)
    w.to(device)

    def fast_nll(x):  # noqa actually used with `eval` below
        model.requires_grad_(False)
        if model.style_encoder is not None:
            style = torch.zeros(1, 1, dtype=torch.int64, device=device)
            utils.print_once("WARNING: using style 0 for input warping, this is set for nonmyopic BO setting.")
        else:
            style = None
        logits = model(x[:, None], y[:, None], x[:, None], style=style, only_return_standard_out=True)
        loss = model.criterion(logits, y[:, None]).squeeze(1)
        return loss

    def true_nll(x): # noqa actually used with `eval` below
        assert model.style_encoder is None, "true nll not implemented for style encoder, see above for an example impl"
        model.requires_grad_(False)
        total_nll = 0.
        for cutoff in range(len(x)):
            logits = model(x[:cutoff, None], y[:cutoff, None], x[cutoff:cutoff + 1, None])
            total_nll = total_nll + model.criterion(logits, y[cutoff:cutoff + 1, None]).squeeze()
        assert len(total_nll.shape) == 0, f"{total_nll.shape=}"
        return total_nll

    def repeated_true_nll(x): # noqa actually used with `eval` below
        assert model.style_encoder is None, "true nll not implemented for style encoder, see above for an example impl"
        model.requires_grad_(False)
        total_nll = 0.
        for i in range(5):
            rs = np.random.RandomState(i)
            shuffle_idx = rs.permutation(len(x))
            x_ = x.clone()[shuffle_idx]
            y_ = y.clone()[shuffle_idx]
            for cutoff in range(len(x)):
                logits = model(x_[:cutoff, None], y_[:cutoff, None], x_[cutoff:cutoff + 1, None])
                total_nll = total_nll + model.criterion(logits, y_[cutoff:cutoff + 1, None]).squeeze()
        assert len(total_nll.shape) == 0, f"{total_nll.shape=}"
        return total_nll

    def repeated_true_100_nll(x): # noqa actually used with `eval` below
        assert model.style_encoder is None, "true nll not implemented for style encoder, see above for an example impl"
        model.requires_grad_(False)
        total_nll = 0.
        for i in range(100):
            rs = np.random.RandomState(i)
            shuffle_idx = rs.permutation(len(x))
            x_ = x.clone()[shuffle_idx]
            y_ = y.clone()[shuffle_idx]
            for cutoff in range(len(x)):
                logits = model(x_[:cutoff, None], y_[:cutoff, None], x_[cutoff:cutoff + 1, None])
                total_nll = total_nll + model.criterion(logits, y_[cutoff:cutoff + 1, None]).squeeze()
        assert len(total_nll.shape) == 0, f"{total_nll.shape=}"
        return total_nll / 100

    def batched_repeated_chunked_true_nll(x): # noqa actually used with `eval` below
        assert model.style_encoder is None, "true nll not implemented for style encoder, see above for an example impl"
        assert len(x.shape) == 2 and len(y.shape) == 1
        model.requires_grad_(False)
        n_features = x.shape[1] if len(x.shape) > 1 else 1
        batch_size = 10

        X = []
        Y = []

        for i in range(batch_size):
            #if i == 0:
            #    shuffle_idx = list(range(len(x)))
            #else:
            rs = np.random.RandomState(i)
            shuffle_idx = rs.permutation(len(x))
            X.append(x.clone()[shuffle_idx])
            Y.append(y.clone()[shuffle_idx])
        X = torch.stack(X, dim=1).view((x.shape[0], batch_size, n_features))
        Y = torch.stack(Y, dim=1).view((x.shape[0], batch_size, 1))

        total_nll = 0.
        batch_indizes = sorted(list(set(np.linspace(0, len(x), 10, dtype=int))))

        for chunk_start, chunk_end in zip(batch_indizes[:-1], batch_indizes[1:]):
            X_cutoff = X[:chunk_start]
            Y_cutoff = Y[:chunk_start]
            X_after_cutoff = X[chunk_start:chunk_end]
            Y_after_cutoff = Y[chunk_start:chunk_end]

            pending_x = X_after_cutoff.reshape(X_after_cutoff.shape[0], batch_size, n_features)  # n_pen x batch_size x n_feat
            observed_x = X_cutoff.reshape(X_cutoff.shape[0], batch_size, n_features)  # n_obs x batch_size x n_feat
            X_tmp = torch.cat((observed_x, pending_x), dim=0)  # (n_obs+n_pen) x batch_size x n_feat

            logits = model((X_tmp, Y_cutoff), single_eval_pos=int(chunk_start))
            total_nll = total_nll + model.criterion(logits, Y_after_cutoff).sum()
        assert len(total_nll.shape) == 0, f"{total_nll.shape=}"
        return total_nll

    def batched_repeated_true_nll(x): # noqa actually used with `eval` below
        assert model.style_encoder is None, "true nll not implemented for style encoder, see above for an example impl"
        model.requires_grad_(False)
        n_features = x.shape[1] if len(x.shape) > 1 else 1
        batch_size = 10

        X = []
        Y = []

        for i in range(batch_size):
            #if i == 0:
            #    shuffle_idx = list(range(len(x)))
            #else:
            rs = np.random.RandomState(i)
            shuffle_idx = rs.permutation(len(x))
            X.append(x.clone()[shuffle_idx])
            Y.append(y.clone()[shuffle_idx])
        X = torch.cat(X, dim=1).reshape((x.shape[0], batch_size, n_features))
        Y = torch.cat(Y, dim=1).reshape((x.shape[0], batch_size, 1))

        total_nll = 0.

        for cutoff in range(0, len(x)):
            X_cutoff = X[:cutoff]
            Y_cutoff = Y[:cutoff]
            X_after_cutoff = X[cutoff:cutoff+1]
            Y_after_cutoff = Y[cutoff:cutoff+1]

            pending_x = X_after_cutoff.reshape(X_after_cutoff.shape[0], batch_size, n_features)  # n_pen x batch_size x n_feat
            observed_x = X_cutoff.reshape(X_cutoff.shape[0], batch_size, n_features)  # n_obs x batch_size x n_feat
            X_tmp = torch.cat((observed_x, pending_x), dim=0)  # (n_obs+n_pen) x batch_size x n_feat

            pad_y = torch.zeros((X_after_cutoff.shape[0], batch_size, 1))  # (n_obs+n_pen) x batch_size
            Y_tmp = torch.cat((Y_cutoff, pad_y), dim=0)

            logits = model((X_tmp, Y_tmp), single_eval_pos=cutoff)
            total_nll = total_nll + model.criterion(logits, Y_after_cutoff).sum()
        assert len(total_nll.shape) == 0, f"{total_nll.shape=}"
        return total_nll

    def one_out_nll(x):  # noqa actually used with `eval` below
        assert model.style_encoder is None, "one out nll not implemented for style encoder, see above for an example impl"
        # x shape: (n, d)
        # iterate over a pre-defined set of
        model.requires_grad_(False)
        #indices = one_out_indices_sampled_per_num_obs[len(x)]
        indices = list(range(x.shape[0]))

        # create batch by moving the one out index to the end
        eval_x = x[indices][None] # shape (1, 10, d)
        eval_y = y[indices][None] # shape (1, 10, 1)
        # all other indices are used for training
        train_x = torch.stack([torch.cat([x[:i], x[i + 1:]]) for i in indices], 1)
        train_y = torch.stack([torch.cat([y[:i], y[i + 1:]]) for i in indices], 1)

        logits = model(train_x, train_y, eval_x)
        return model.criterion(logits, eval_y).squeeze(0)

    def subset_nll(x):  # noqa actually used with `eval` below
        assert model.style_encoder is None, "subset nll not implemented for style encoder, see above for an example impl"
        # x shape: (n, d)
        # iterate over a pre-defined set of
        model.requires_grad_(False)
        eval_indices = torch.tensor(subsets[len(x)])
        train_indices = torch.tensor(neg_subsets[len(x)])

        # batch by using all eval_indices
        eval_x = x[eval_indices.flatten()].view(eval_indices.shape + (-1,)) # shape (10, n//2, d)
        eval_y = y[eval_indices.flatten()].view(eval_indices.shape + (-1,)) # shape (10, n//2, 1)
        # all other indices are used for training
        train_x = x[train_indices.flatten()].view(train_indices.shape + (-1,)) # shape (10, n//2, d)
        train_y = y[train_indices.flatten()].view(train_indices.shape + (-1,)) # shape (10, n//2, 1)

        logits = model(train_x.transpose(0, 1), train_y.transpose(0, 1), eval_x.transpose(0, 1))
        return model.criterion(logits, eval_y.transpose(0, 1))

    if opt_method == "log_vs_nolog":
        log_vs_nonlog(x, w, eval(nll_type + '_nll'),
            ignore_prior=True, # true_nll=repeated_true_100_nll,
            **kwargs)
    elif opt_method == "lbfgs":
        fit_lbfgs_with_restarts(
            x, w, eval(nll_type + '_nll'),
            ignore_prior=True, old_solution=old_solution, # true_nll=repeated_true_100_nll,
            **kwargs)
    elif opt_method == "lbfgs_w_prior":
        fit_lbfgs_with_restarts(
            x, w, eval(nll_type + '_nll'),
            ignore_prior=False, old_solution=old_solution,  # true_nll=repeated_true_100_nll,
            **kwargs)
    else:
        raise ValueError(opt_method)

    return w
