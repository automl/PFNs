import torch
from torch.distributions import Kumaraswamy


def exp_in_prev_range(x,factor):
    mini, maxi = x.min(0)[0], x.max(0)[0]
    expx = (factor*x).exp()
    expx_01 = (expx - expx.min(0)[0]) / (expx.max(0)[0] - expx.min(0)[0])
    return expx_01 * (maxi - mini) + mini


@torch.no_grad()
def get_batch(*args, hyperparameters, get_batch, **kwargs):
    """
    This `get_batch` can be used to wrap another `get_batch` and apply a Kumaraswamy transform to the input.
    The x's have to be in [0,1] for this to work!
    """
    returns = get_batch(*args, hyperparameters=hyperparameters, **kwargs)

    input_warping_type = hyperparameters.get('input_warping_type', 'kumar')
    # controls what part of the batch ('x', 'y' or 'xy') to apply the warping to
    input_warping_groups = hyperparameters.get('input_warping_groups', 'x')
    # whether to norm inputs between 0 and 1 before warping, as warping is only possible in that range.
    input_warping_norm = hyperparameters.get('input_warping_norm', False)
    use_icdf = hyperparameters.get('input_warping_use_icdf', False)

    def norm_to_0_1(x):
        eps = .00001
        maxima = torch.max(x, 0)[0]
        minima = torch.min(x, 0)[0]
        normed_x = (x - minima) / (maxima - minima + eps)

        def denorm(normed_x):
            return normed_x * (maxima - minima + eps) + minima

        return normed_x, denorm

    def warp_input(x):
        if input_warping_norm:
            x, denorm = norm_to_0_1(x)

        if input_warping_type == 'kumar':
            if 'input_warping_c_std' in hyperparameters:
                assert 'input_warping_c0_std' not in hyperparameters and 'input_warping_c1_std' not in hyperparameters
                hyperparameters['input_warping_c0_std'] = hyperparameters['input_warping_c_std']
                hyperparameters['input_warping_c1_std'] = hyperparameters['input_warping_c_std']
            inside = 0
            while not inside:
                c1 = (torch.randn(*x.shape[1:], device=x.device) * hyperparameters.get('input_warping_c1_std', .75)).exp()
                c0 = (torch.randn(*x.shape[1:], device=x.device) * hyperparameters.get('input_warping_c0_std', .75)).exp()
                if not hyperparameters.get('input_warping_in_range', False):
                    inside = True
                elif (c1 < 10).all() and (c1 > 0).all() and (c0 < 10).all() and (c0 > 0).all():
                    inside = True
                else:
                    inside -= 1
                    if inside < -100:
                        print('It seems that the input warping is not working.')
            if c1_v := hyperparameters.get('fix_input_warping_c1', False):
                c1[:] = c1_v
            if c0_v := hyperparameters.get('fix_input_warping_c0', False):
                c0[:] = c0_v
            if hyperparameters.get('verbose', False):
                print(f'c1: {c1}, c0: {c0}')
            k = Kumaraswamy(concentration1=c1, concentration0=c0)
            x_transformed = k.icdf(x) if use_icdf else k.cdf(x)
        elif input_warping_type == 'exp':
            transform_likelihood = hyperparameters.get('input_warping_transform_likelihood', 0.2)
            to_be_transformed = torch.rand_like(x[0,0]) < transform_likelihood
            transform_factors = torch.rand_like(x[0,0]) * hyperparameters.get('input_warping_transform_factor', 1.)
            log_direction = torch.rand_like(x[0,0]) < 0.5
            exp_x = exp_in_prev_range(x, transform_factors)
            minus_exp_x = 1.-exp_in_prev_range(1.-x, transform_factors)
            exp_x = torch.where(log_direction, exp_x, minus_exp_x)
            x_transformed = torch.where(to_be_transformed[None,None,:], exp_x, x)
        elif input_warping_type is None or input_warping_type == 'none':
            x_transformed = x
        else:
            raise ValueError(f"Unknown input_warping_type: {input_warping_type}")

        if input_warping_norm:
            x_transformed = denorm(x_transformed)
        return x_transformed

    if 'x' in input_warping_groups:
        returns.x = warp_input(returns.x)
    if 'y' in input_warping_groups:
        returns.y = warp_input(returns.y)
        returns.target_y = warp_input(returns.target_y)

    return returns


