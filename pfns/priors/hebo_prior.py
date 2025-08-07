import warnings

import botorch
import gpytorch
import torch
from botorch.exceptions import InputDataWarning

from gpytorch.means import ZeroMean
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior, UniformPrior

from ..utils import default_device, to_tensor

from . import utils
from .prior import Batch


def constraint_based_on_distribution_support(
    prior: torch.distributions.Distribution, device, sample_from_path
):
    if sample_from_path:
        return None

    if hasattr(prior.support, "upper_bound"):
        return gpytorch.constraints.Interval(
            to_tensor(prior.support.lower_bound, device=device),
            to_tensor(prior.support.upper_bound, device=device),
        )
    else:
        return gpytorch.constraints.GreaterThan(
            to_tensor(prior.support.lower_bound, device=device)
        )


loaded_things = {}


def torch_load(path):
    """
    Cached torch load. Caution: This does not copy the output but keeps pointers.
    That means, if you modify the output, you modify the output of later calls to this function with the same args.
    :param path:
    :return:
    """
    if path not in loaded_things:
        print(f"loading {path}")
        with open(path, "rb") as f:
            loaded_things[path] = torch.load(f)
    return loaded_things[path]


def _compute_gamma_params(
    mean: float | None = None,
    std: float | None = None,
    concentration: float | None = None,
    rate: float | None = None,
) -> tuple[float, float]:
    """Helper function to compute gamma distribution parameters.
    Either provide mean and std, or concentration and rate.

    Args:
        mean: Mean of the gamma distribution (if using mean/std parameterization)
        std: Standard deviation of the gamma distribution (if using mean/std parameterization)
        concentration: Shape/concentration parameter (α) of the gamma distribution
        rate: Rate parameter (β) of the gamma distribution

    Returns:
        Tuple of (concentration, rate)
    """
    if concentration is not None and rate is not None:
        return concentration, rate
    elif mean is not None and std is not None:
        # For Gamma distribution:
        # mean = α/β
        # variance = α/β² = std²
        # Therefore:
        # α = mean²/variance = (mean/std)²
        # β = mean/variance = mean/(std²)
        variance = std * std
        concentration = mean * mean / variance
        rate = mean / variance
        return concentration, rate
    else:
        raise ValueError("Must provide either (mean, std) or (concentration, rate)")


def to_random_module_no_copy(module) -> gpytorch.Module:
    random_module_cls = type(
        "_Random" + module.__class__.__name__,
        (gpytorch.module.RandomModuleMixin, module.__class__),
        {},
    )
    module.__class__ = random_module_cls  # hack

    for mname, child in module.named_children():
        if isinstance(child, gpytorch.Module):
            setattr(module, mname, to_random_module_no_copy(child))
    return module


def get_model(x, y, hyperparameters: dict, sample=True, no_deepcopy=True):
    sample_from_path = hyperparameters.get("sample_from_extra_prior", None)
    device = x.device
    num_features = x.shape[-1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
    )
    likelihood.register_prior(
        "noise_prior",
        LogNormalPrior(
            torch.tensor(
                hyperparameters.get("hebo_noise_logmean", -4.63), device=device
            ),
            torch.tensor(hyperparameters.get("hebo_noise_std", 0.5), device=device),
        ),
        "noise",
    )

    # Handle lengthscale prior parameters
    lengthscale_concentration, lengthscale_rate = _compute_gamma_params(
        mean=hyperparameters.get("lengthscale_mean", None),
        std=hyperparameters.get("lengthscale_std", None),
        concentration=hyperparameters.get("lengthscale_concentration", None),
        rate=hyperparameters.get("lengthscale_rate", None),
    )

    # print(f'{lengthscale_concentration=}, {lengthscale_rate=}')

    lengthscale_prior = GammaPrior(
        torch.tensor(lengthscale_concentration, device=device),
        torch.tensor(lengthscale_rate, device=device),
    )

    covar_module = gpytorch.kernels.MaternKernel(
        nu=3 / 2,
        ard_num_dims=num_features,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=constraint_based_on_distribution_support(
            lengthscale_prior, device, sample_from_path
        ),
    )

    # Handle outputscale prior parameters
    outputscale_concentration, outputscale_rate = _compute_gamma_params(
        mean=hyperparameters.get("outputscale_mean", None),
        std=hyperparameters.get("outputscale_std", None),
        concentration=hyperparameters.get("outputscale_concentration", None),
        rate=hyperparameters.get("outputscale_rate", None),
    )

    # print(f'{outputscale_concentration=}, {outputscale_rate=}')

    outputscale_prior = GammaPrior(
        concentration=torch.tensor(outputscale_concentration, device=device),
        rate=torch.tensor(outputscale_rate, device=device),
    )

    # print(f'{outputscale_prior.mean=}, {outputscale_prior.variance=}')

    covar_module = gpytorch.kernels.ScaleKernel(
        covar_module,
        outputscale_prior=outputscale_prior,
        outputscale_constraint=constraint_based_on_distribution_support(
            outputscale_prior, device, sample_from_path
        ),
    )

    if torch.rand(1).item() < float(hyperparameters.get("add_linear_kernel", True)):
        # ORIG DIFF: added priors for variance and outputscale of linear kernel
        var_prior = UniformPrior(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )
        out_prior = UniformPrior(
            torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        )
        lincovar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(
                variance_prior=var_prior,
                variance_constraint=constraint_based_on_distribution_support(
                    var_prior, device, sample_from_path
                ),
            ),
            outputscale_prior=out_prior,
            outputscale_constraint=constraint_based_on_distribution_support(
                out_prior, device, sample_from_path
            ),
        )
        covar_module = covar_module + lincovar_module

    # assume mean 0 always!
    if len(y.shape) < len(x.shape):
        y = y.unsqueeze(-1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)
        model = botorch.models.SingleTaskGP(
            x,
            y,
            likelihood=likelihood,
            covar_module=covar_module,
        )
    model.mean_module = ZeroMean(x.shape[:-2])
    model.to(device)
    likelihood.to(device)

    if sample:
        if no_deepcopy:
            # code same as model.pyro_sample_from_prior(), just without the deepcopy
            from gpytorch.module import _pyro_sample_from_prior

            model = to_random_module_no_copy(model)
            _pyro_sample_from_prior(module=model, memo=None, prefix="")
        else:
            model = model.pyro_sample_from_prior()
        if sample_from_path:
            parameter_sample_distribution = torch_load(
                sample_from_path
            )  # dict with entries for each parameter
            idx_for_len = {}
            for (
                parameter_name,
                parameter_values,
            ) in parameter_sample_distribution.items():
                assert len(parameter_values.shape) == 1
                try:
                    p = eval(parameter_name)
                    if len(parameter_values) in idx_for_len:
                        idx = idx_for_len[len(parameter_values)].view(p.shape)
                    else:
                        idx = torch.randint(len(parameter_values), p.shape)
                        idx_for_len[len(parameter_values)] = idx
                    new_sample = parameter_values[idx].to(device).view(p.shape)  # noqa
                    assert new_sample.shape == p.shape
                    with torch.no_grad():
                        p.data = new_sample
                except AttributeError:
                    utils.print_once(
                        f"could not find parameter {parameter_name} in model for `sample_from_extra_prior`"
                    )
            model.requires_grad_(False)
            likelihood.requires_grad_(False)
        return model, model.likelihood
    else:
        assert not (hyperparameters.get("sigmoid", False)) and not (
            hyperparameters.get("y_minmax_norm", False)
        ), "Sigmoid and y_minmax_norm can only be used to sample models..."
        return model, likelihood


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    device=default_device,
    hyperparameters=None,
    batch_size_per_gp_sample=None,
    single_eval_pos=None,
    fix_to_range=None,
    equidistant_x=False,
    verbose=False,
    **kwargs,
):
    """
    This function is very similar to the equivalent in .fast_gp. The only difference is that this function operates over
    a mixture of GP priors.
    :param batch_size:
    :param seq_len:
    :param num_features:
    :param device:
    :param hyperparameters:
    :param for_regression:
    :return:
    """
    device = "cpu"
    hyperparameters = hyperparameters or {}
    with gpytorch.settings.fast_computations(
        *hyperparameters.get("fast_computations", (True, True, True))
    ):
        batch_size_per_gp_sample = batch_size_per_gp_sample or max(batch_size // 4, 1)
        assert (
            batch_size % batch_size_per_gp_sample == 0
        ), f"{batch_size} % {batch_size_per_gp_sample} != 0"

        total_num_candidates = batch_size * (2 ** (fix_to_range is not None))
        num_candidates = batch_size_per_gp_sample * (2 ** (fix_to_range is not None))
        unused_feature_likelihood = hyperparameters.get(
            "unused_feature_likelihood", False
        )
        if equidistant_x:
            assert num_features == 1
            assert not unused_feature_likelihood
            x = (
                torch.linspace(0, 1.0, seq_len)
                .unsqueeze(0)
                .repeat(total_num_candidates, 1)
                .unsqueeze(-1)
            )
        else:
            if hyperparameters["x_sampler"] == "uni":
                x = torch.rand(
                    total_num_candidates, seq_len, num_features, device=device
                )
            elif hyperparameters["x_sampler"] == "normal":
                unnormalized_x = torch.randn(
                    total_num_candidates, seq_len, num_features, device=device
                )
                # Normalize each feature across the seq_len dimension to be within [0,1]
                # Reshape to make operations easier
                reshaped_x = unnormalized_x.transpose(
                    1, 2
                )  # [batch, features, seq_len]
                # Get min and max values for each feature in each batch
                min_vals, _ = torch.min(reshaped_x, dim=2, keepdim=True)
                max_vals, _ = torch.max(reshaped_x, dim=2, keepdim=True)
                # Handle the case where min == max to avoid division by zero
                denom = max_vals - min_vals
                denom[denom == 0] = 1.0  # Set denominator to 1 where max == min
                # Normalize to [0,1]
                normalized_x = (reshaped_x - min_vals) / denom
                # Reshape back to original format
                x = normalized_x.transpose(1, 2)  # [batch, seq_len, features]
            else:
                raise NotImplementedError(
                    f"Your x_sampler ({hyperparameters['x_sampler']}) setting is not implemented."
                )

        samples = []
        samples_wo_noise = []
        for i in range(0, total_num_candidates, num_candidates):
            local_x = x[i : i + num_candidates]
            if unused_feature_likelihood:
                r = torch.rand(num_features)
                unused_feature_mask = r < unused_feature_likelihood
                if unused_feature_mask.all():
                    unused_feature_mask[r.argmin()] = False
                used_local_x = local_x[..., ~unused_feature_mask]
            else:
                used_local_x = local_x
            get_model_and_likelihood = lambda used_local_x: get_model(  # noqa: E731
                used_local_x,
                torch.randn(num_candidates, x.shape[1], device=device),
                hyperparameters,
            )
            model, likelihood = get_model_and_likelihood(used_local_x)
            if verbose:
                print(
                    list(model.named_parameters()),
                    (
                        (
                            list(model.input_transform.named_parameters()),
                            model.input_transform.concentration1,
                            model.input_transform.concentration0,
                        )
                        if model.input_transform is not None
                        else None
                    ),
                )

            # trained_model = ExactGPModel(train_x, train_y, likelihood).cuda()
            # trained_model.eval()
            successful_sample = 0
            throwaway_share = 0.0
            while successful_sample < 1:
                with gpytorch.settings.prior_mode(True):
                    # print(x.device, device, f'{model.covar_module.base_kernel.lengthscale=}, {model.covar_module.base_kernel.lengthscale.device=}')
                    d = model(used_local_x)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            sample_wo_noise = d.sample()
                            d = likelihood(sample_wo_noise)
                    except (RuntimeError, ValueError) as e:
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood(used_local_x)
                        if successful_sample < -100:
                            print(
                                f"Could not sample from model {i} after {successful_sample} attempts. {e}"
                            )
                            raise e
                        continue
                    sample = d.sample()  # bs_per_gp_s x T
                    if fix_to_range is None:
                        # for k, v in model.named_parameters(): print(k,v)
                        samples.append(sample)
                        samples_wo_noise.append(sample_wo_noise)
                        break
                    smaller_mask = sample < fix_to_range[0]
                    larger_mask = sample >= fix_to_range[1]
                    in_range_mask = ~(smaller_mask | larger_mask).any(1)
                    throwaway_share += (
                        ~in_range_mask[:batch_size_per_gp_sample]
                    ).sum() / batch_size_per_gp_sample
                    if in_range_mask.sum() < batch_size_per_gp_sample:
                        successful_sample -= 1
                        if successful_sample < 100:
                            print(
                                "Please change hyper-parameters (e.g. decrease outputscale_mean) it"
                                "seems like the range is set to tight for your hyper-parameters."
                            )
                        continue

                    x[i : i + batch_size_per_gp_sample] = local_x[in_range_mask][
                        :batch_size_per_gp_sample
                    ]
                    sample = sample[in_range_mask][:batch_size_per_gp_sample]
                    samples.append(sample)
                    samples_wo_noise.append(sample_wo_noise)
                    successful_sample = True

        if torch.rand(1).item() < 0.0001:
            print(
                "throwaway share",
                throwaway_share / (batch_size // batch_size_per_gp_sample),
            )

        # print(f'{[s.shape for s in samples]=}, {x.shape=}')

        # print(f'took {time.time() - start}')
        sample = torch.cat(samples, 0)[..., None].float()
        sample_wo_noise = torch.cat(samples_wo_noise, 0)[..., None].float()
        x = x.view(-1, batch_size, seq_len, num_features)[0]
        # TODO think about enabling the line below
        # sample = sample - sample[0, :].unsqueeze(0).expand(*sample.shape)
        assert x.shape[:2] == sample.shape[:2], f"{x.shape} != {sample.shape}"
    return Batch(
        x=x.float(),
        y=sample,
        target_y=(
            sample
            if hyperparameters.get("observation_noise", True)
            else sample_wo_noise
        ),
    )
