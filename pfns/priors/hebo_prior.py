import time
import functools
import random
import math
import traceback
import warnings

import numpy as np
import torch
from torch import nn
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, UniformPrior, LogNormalPrior
from gpytorch.means import ZeroMean
from botorch.models.transforms.input import *
from gpytorch.constraints import GreaterThan

from . import utils
from ..utils import default_device, to_tensor
from .prior import Batch
from .utils import get_batch_to_dataloader

class Warp(gpytorch.Module):
    r"""A transform that uses learned input warping functions.

    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.

    for each output in batched multi-output and multi-task models.

    For now, ModelListGPs should be used to learn independent warping
    functions for each output.
    """

    _min_concentration_level = 1e-4

    def __init__(
        self,
        indices: List[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        eps: float = 1e-7,
        concentration1_prior: Optional[Prior] = None,
        concentration0_prior: Optional[Prior] = None,
        batch_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize transform.

        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            concentration1_prior: A prior distribution on the concentration1 parameter
                of the Kumaraswamy distribution.
            concentration0_prior: A prior distribution on the concentration0 parameter
                of the Kumaraswamy distribution.
            batch_shape: The batch shape.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse
        self.batch_shape = batch_shape or torch.Size([])
        self._X_min = eps
        self._X_range = 1 - 2 * eps
        if len(self.batch_shape) > 0:
            # Note: this follows the gpytorch shape convention for lengthscales
            # There is ongoing discussion about the extra `1`.
            # https://github.com/cornellius-gp/gpytorch/issues/1317
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape
        for i in (0, 1):
            p_name = f"concentration{i}"
            self.register_parameter(
                p_name,
                nn.Parameter(torch.full(batch_shape + self.indices.shape, 1.0)),
            )
        if concentration0_prior is not None:
            def closure(m):
                #print(m.concentration0)
                return m.concentration0
            self.register_prior(
                "concentration0_prior",
                concentration0_prior,
                closure,
                lambda m, v: m._set_concentration(i=0, value=v),
            )
        if concentration1_prior is not None:
            def closure(m):
                #print(m.concentration1)
                return m.concentration1
            self.register_prior(
                "concentration1_prior",
                concentration1_prior,
                closure,
                lambda m, v: m._set_concentration(i=1, value=v),
            )
        for i in (0, 1):
            p_name = f"concentration{i}"
            constraint = GreaterThan(
                self._min_concentration_level,
                transform=None,
                # set the initial value to be the identity transformation
                initial_value=1.0,
            )
            self.register_constraint(param_name=p_name, constraint=constraint)

    def _set_concentration(self, i: int, value: Union[float, Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.concentration0)
        self.initialize(**{f"concentration{i}": value})

    def _transform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy CDF.

        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.

        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor of transformed
                inputs.
        """
        X_tf = expand_and_copy_tensor(X=X, batch_shape=self.batch_shape)
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        # normalize to [eps, 1-eps]
        X_tf[..., self.indices] = k.cdf(
            torch.clamp(
                X_tf[..., self.indices] * self._X_range + self._X_min,
                self._X_min,
                1.0 - self._X_min,
            )
        )
        return X_tf

    def _untransform(self, X: Tensor) -> Tensor:
        r"""Warp the inputs through the Kumaraswamy inverse CDF.

        Args:
            X: A `input_batch_shape x batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `input_batch_shape x batch_shape x n x d`-dim tensor of transformed
                inputs.
        """
        if len(self.batch_shape) > 0:
            if self.batch_shape != X.shape[-2 - len(self.batch_shape) : -2]:
                raise BotorchTensorDimensionError(
                    "The right most batch dims of X must match self.batch_shape: "
                    f"({self.batch_shape})."
                )
        X_tf = X.clone()
        k = Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        # unnormalize from [eps, 1-eps] to [0,1]
        X_tf[..., self.indices] = (
            (k.icdf(X_tf[..., self.indices]) - self._X_min) / self._X_range
        ).clamp(0.0, 1.0)
        return X_tf

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        return self._untransform(X) if self.reverse else self._transform(X)

    def untransform(self, X: Tensor) -> Tensor:
        r"""Un-transform the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        return self._transform(X) if self.reverse else self._untransform(X)

    def equals(self, other: InputTransform) -> bool:
        r"""Check if another input transform is equivalent.

        Note: The reason that a custom equals method is defined rather than
        defining an __eq__ method is because defining an __eq__ method sets
        the __hash__ method to None. Hashing modules is currently used in
        pytorch. See https://github.com/pytorch/pytorch/issues/7733.

        Args:
            other: Another input transform.

        Returns:
            A boolean indicating if the other transform is equivalent.
        """
        other_state_dict = other.state_dict()
        return (
            type(self) == type(other)
            and (self.transform_on_train == other.transform_on_train)
            and (self.transform_on_eval == other.transform_on_eval)
            and (self.transform_on_fantasize == other.transform_on_fantasize)
            and all(
                torch.allclose(v, other_state_dict[k].to(v))
                for k, v in self.state_dict().items()
            )
        )

    def preprocess_transform(self, X: Tensor) -> Tensor:
        r"""Apply transforms for preprocessing inputs.

        The main use cases for this method are 1) to preprocess training data
        before calling `set_train_data` and 2) preprocess `X_baseline` for noisy
        acquisition functions so that `X_baseline` is "preprocessed" with the
        same transformations as the cached training inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of (transformed) inputs.
        """
        if self.transform_on_train:
            # We need to disable learning of bounds here.
            # See why: https://github.com/pytorch/botorch/issues/1078.
            if hasattr(self, "learn_bounds"):
                learn_bounds = self.learn_bounds
                self.learn_bounds = False
                result = self.transform(X)
                self.learn_bounds = learn_bounds
                return result
            else:
                return self.transform(X)
        return X

    def forward(self, X: Tensor) -> Tensor:
        r"""Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n' x d`-dim tensor of transformed inputs.
        """
        if self.training:
            if self.transform_on_train:
                return self.transform(X)
        elif self.transform_on_eval:
            if fantasize.off() or self.transform_on_fantasize:
                return self.transform(X)
        return X

def constraint_based_on_distribution_support(prior: torch.distributions.Distribution, device, sample_from_path):
    if sample_from_path:
        return None

    if hasattr(prior.support, 'upper_bound'):
        return gpytorch.constraints.Interval(to_tensor(prior.support.lower_bound,device=device),
                                             to_tensor(prior.support.upper_bound,device=device))
    else:
        return gpytorch.constraints.GreaterThan(to_tensor(prior.support.lower_bound,device=device))


loaded_things = {}
def torch_load(path):
    '''
    Cached torch load. Caution: This does not copy the output but keeps pointers.
    That means, if you modify the output, you modify the output of later calls to this function with the same args.
    :param path:
    :return:
    '''
    if path not in loaded_things:
        print(f'loading {path}')
        with open(path, 'rb') as f:
            loaded_things[path] = torch.load(f)
    return loaded_things[path]


def get_model(x, y, hyperparameters: dict, sample=True):
    sample_from_path = hyperparameters.get('sample_from_extra_prior', None)
    device = x.device
    num_features = x.shape[-1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    likelihood.register_prior("noise_prior",
                              LogNormalPrior(torch.tensor(hyperparameters.get('hebo_noise_logmean',-4.63), device=device),
                                             torch.tensor(hyperparameters.get('hebo_noise_std', 0.5), device=device)
                                             ),
                              "noise")
    lengthscale_prior = \
        GammaPrior(
            torch.tensor(hyperparameters['lengthscale_concentration'], device=device),
            torch.tensor(hyperparameters['lengthscale_rate'], device=device))\
        if hyperparameters.get('lengthscale_concentration', None) else\
        UniformPrior(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    covar_module = gpytorch.kernels.MaternKernel(nu=3 / 2, ard_num_dims=num_features,
                                                 lengthscale_prior=lengthscale_prior,
                                                 lengthscale_constraint=\
                                                     constraint_based_on_distribution_support(lengthscale_prior, device, sample_from_path))
    # ORIG DIFF: orig lengthscale has no prior
    #covar_module.register_prior("lengthscale_prior",
                                #UniformPrior(.000000001, 1.),
                                #GammaPrior(concentration=hyperparameters.get('lengthscale_concentration', 1.),
                                #           rate=hyperparameters.get('lengthscale_rate', .1)),
                                # skewness is controllled by concentration only, want somthing like concetration in [0.1,1.], rate around [.05,1] seems reasonable
                                #"lengthscale")
    outputscale_prior = \
        GammaPrior(concentration=hyperparameters.get('outputscale_concentration', .5),
                   rate=hyperparameters.get('outputscale_rate', 1.))
    covar_module = gpytorch.kernels.ScaleKernel(covar_module, outputscale_prior=outputscale_prior,
                                                outputscale_constraint=constraint_based_on_distribution_support(outputscale_prior, device, sample_from_path))

    if random.random() < float(hyperparameters.get('add_linear_kernel', True)):
        # ORIG DIFF: added priors for variance and outputscale of linear kernel
        var_prior = UniformPrior(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        out_prior = UniformPrior(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        lincovar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(
            variance_prior=var_prior,
            variance_constraint=constraint_based_on_distribution_support(var_prior,device,sample_from_path),
        ),
            outputscale_prior=out_prior,
            outputscale_constraint=constraint_based_on_distribution_support(out_prior,device,sample_from_path),
        )
        covar_module = covar_module + lincovar_module

    if hyperparameters.get('hebo_warping', False):
        # initialize input_warping transformation
        warp_tf = Warp(
            indices=list(range(num_features)),
            # use a prior with median at 1.
            # when a=1 and b=1, the Kumaraswamy CDF is the identity function
            concentration1_prior=LogNormalPrior(torch.tensor(0.0, device=device), torch.tensor(hyperparameters.get('hebo_input_warping_c1_std',.75), device=device)),
            concentration0_prior=LogNormalPrior(torch.tensor(0.0, device=device), torch.tensor(hyperparameters.get('hebo_input_warping_c0_std',.75), device=device)),
        )
    else:
        warp_tf = None
    # assume mean 0 always!
    if len(y.shape) < len(x.shape):
        y = y.unsqueeze(-1)
    model = botorch.models.SingleTaskGP(x, y, likelihood, covar_module=covar_module, input_transform=warp_tf)
    model.mean_module = ZeroMean(x.shape[:-2])
    model.to(device)
    likelihood.to(device)

    if sample:
        model = model.pyro_sample_from_prior()
        if sample_from_path:
            parameter_sample_distribution = torch_load(sample_from_path) # dict with entries for each parameter
            idx_for_len = {}
            for parameter_name, parameter_values in parameter_sample_distribution.items():
                assert len(parameter_values.shape) == 1
                try:
                    p = eval(parameter_name)
                    if len(parameter_values) in idx_for_len:
                        idx = idx_for_len[len(parameter_values)].view(p.shape)
                    else:
                        idx = torch.randint(len(parameter_values), p.shape)
                        idx_for_len[len(parameter_values)] = idx
                    new_sample = parameter_values[idx].to(device).view(p.shape) # noqa
                    assert new_sample.shape == p.shape
                    with torch.no_grad():
                        p.data = new_sample
                except AttributeError:
                    utils.print_once(f'could not find parameter {parameter_name} in model for `sample_from_extra_prior`')
            model.requires_grad_(False)
            likelihood.requires_grad_(False)
        return model, model.likelihood
    else:
        assert not(hyperparameters.get('sigmoid', False)) and not(hyperparameters.get('y_minmax_norm', False)), "Sigmoid and y_minmax_norm can only be used to sample models..."
        return model, likelihood


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              batch_size_per_gp_sample=None, single_eval_pos=None,
              fix_to_range=None, equidistant_x=False, verbose=False, **kwargs):
    '''
    This function is very similar to the equivalent in .fast_gp. The only difference is that this function operates over
    a mixture of GP priors.
    :param batch_size:
    :param seq_len:
    :param num_features:
    :param device:
    :param hyperparameters:
    :param for_regression:
    :return:
    '''
    hyperparameters = hyperparameters or {}
    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))):
        batch_size_per_gp_sample = (batch_size_per_gp_sample or max(batch_size // 4,1))
        assert batch_size % batch_size_per_gp_sample == 0

        total_num_candidates = batch_size*(2**(fix_to_range is not None))
        num_candidates = batch_size_per_gp_sample * (2**(fix_to_range is not None))
        unused_feature_likelihood = hyperparameters.get('unused_feature_likelihood', False)
        if equidistant_x:
            assert num_features == 1
            assert not unused_feature_likelihood
            x = torch.linspace(0,1.,seq_len).unsqueeze(0).repeat(total_num_candidates,1).unsqueeze(-1)
        else:
            x = torch.rand(total_num_candidates, seq_len, num_features, device=device)
        samples = []
        samples_wo_noise = []
        for i in range(0, total_num_candidates, num_candidates):
            local_x = x[i:i+num_candidates]
            if unused_feature_likelihood:
                r = torch.rand(num_features)
                unused_feature_mask = r < unused_feature_likelihood
                if unused_feature_mask.all():
                    unused_feature_mask[r.argmin()] = False
                used_local_x = local_x[...,~unused_feature_mask]
            else:
                used_local_x = local_x
            get_model_and_likelihood = lambda: get_model(used_local_x, torch.zeros(num_candidates,x.shape[1], device=device), hyperparameters)
            model, likelihood = get_model_and_likelihood()
            if verbose: print(list(model.named_parameters()),
                              (list(model.input_transform.named_parameters()), model.input_transform.concentration1, model.input_transform.concentration0)
                                  if model.input_transform is not None else None,
                              )

            # trained_model = ExactGPModel(train_x, train_y, likelihood).cuda()
            # trained_model.eval()
            successful_sample = 0
            throwaway_share = 0.
            while successful_sample < 1:
                with gpytorch.settings.prior_mode(True):
                    #print(x.device, device, f'{model.covar_module.base_kernel.lengthscale=}, {model.covar_module.base_kernel.lengthscale.device=}')
                    d = model(used_local_x)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            sample_wo_noise = d.sample()
                            d = likelihood(sample_wo_noise)
                    except (RuntimeError, ValueError) as e:
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood()
                        if successful_sample < -100:
                            print(f'Could not sample from model {i} after {successful_sample} attempts. {e}')
                            raise e
                        continue
                    sample = d.sample() # bs_per_gp_s x T
                    if fix_to_range is None:
                        #for k, v in model.named_parameters(): print(k,v)
                        samples.append(sample.transpose(0, 1))
                        samples_wo_noise.append(sample_wo_noise.transpose(0, 1))
                        break
                    smaller_mask = sample < fix_to_range[0]
                    larger_mask = sample >= fix_to_range[1]
                    in_range_mask = ~ (smaller_mask | larger_mask).any(1)
                    throwaway_share += (~in_range_mask[:batch_size_per_gp_sample]).sum()/batch_size_per_gp_sample
                    if in_range_mask.sum() < batch_size_per_gp_sample:
                        successful_sample -= 1
                        if successful_sample < 100:
                            print("Please change hyper-parameters (e.g. decrease outputscale_mean) it"
                                  "seems like the range is set to tight for your hyper-parameters.")
                        continue

                    x[i:i+batch_size_per_gp_sample] = local_x[in_range_mask][:batch_size_per_gp_sample]
                    sample = sample[in_range_mask][:batch_size_per_gp_sample]
                    samples.append(sample.transpose(0, 1))
                    samples_wo_noise.append(sample_wo_noise.transpose(0, 1))
                    successful_sample = True

        if random.random() < .01:
            print('throwaway share', throwaway_share/(batch_size//batch_size_per_gp_sample))

        #print(f'took {time.time() - start}')
        sample = torch.cat(samples, 1)[...,None]
        sample_wo_noise = torch.cat(samples_wo_noise, 1)[...,None]
        x = x.view(-1,batch_size,seq_len,num_features)[0]
        # TODO think about enabling the line below
        #sample = sample - sample[0, :].unsqueeze(0).expand(*sample.shape)
        x = x.transpose(0,1)
        assert x.shape[:2] == sample.shape[:2]
    return Batch(x=x, y=sample, target_y=sample if hyperparameters.get('observation_noise', True) else sample_wo_noise)

DataLoader = get_batch_to_dataloader(get_batch)
