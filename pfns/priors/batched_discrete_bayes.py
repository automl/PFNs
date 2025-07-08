import math
from abc import ABCMeta, abstractmethod
from numbers import Real

import torch
import torch.nn.functional as F

from pfns.priors import Batch
from pfns.utils import to_tensor


class DiscretePrior(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, weights: torch.Tensor):
        """
        :param weights: (num_latent_options, )
        """
        super().__init__()
        self.register_buffer("weights", weights)

    def sample_latent_index(self, num_samples=1):
        # cumweights = torch.cumsum(self.weights, 0).to(torch.double)
        # cumweights = cumweights / cumweights[-1]
        # return torch.searchsorted(cumweights, torch.rand(num_samples, dtype=torch.double, device=self.device))
        return torch.multinomial(self.weights, num_samples, replacement=True)

    @property
    def prior_logprobs(self):
        probs = self.weights / self.weights.sum()
        return probs.log()

    def __len__(self):
        return len(self.weights)

    @property
    def device(self):
        return self.weights.device

    def to_latent_vector(self, latent_value: torch.Tensor):
        """
        Transforms a latent values that can either be of shape (,) or (#latent options, ) to (#latent options, )
        :param latent_value: tensor[latent options] or tensor[]
        :return:
        """
        if len(latent_value.shape) == 0:
            l = latent_value[None]
            return l.expand(len(self))
        else:
            return latent_value

    @abstractmethod
    def sample_xy(self, num_samples=1):
        """
        Sample from p(x, y) num_samples times, from the latent options
        Return x: tensor, y: tensor
        """
        pass

    @abstractmethod
    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        """
        pass

    def mean_y(self, x: torch.Tensor):
        """
        This computes E[y|x,latent] -> tensor[batch size, # latent options], where x is batched.
        This function is only used for regression.
        """
        raise NotImplementedError()

    def probs_y(self, x: torch.Tensor):
        """
        This computes p(y|x,latent) -> tensor[batch size, #latent options, #classes]
        This function is only used for classification.
        """
        raise NotImplementedError


def x_sampler(num_samples):
    return torch.rand(num_samples, 1) * 2 - 1


x_sampler.logprob = lambda x: torch.full(x.shape, math.log(0.5))


class Step(DiscretePrior):
    def __init__(
        self,
        weights: torch.Tensor,
        y_offsets: float | torch.Tensor = 0.0,
        x_offsets: float | torch.Tensor = 0.0,
        amplitudes: float | torch.Tensor = 1.0,
        noise_std: float = 0.1,
        noise_type: str = "normal",
    ):
        super().__init__(weights)
        self.y_offsets = to_tensor(y_offsets)
        self.x_offsets = to_tensor(x_offsets)
        self.amplitudes = to_tensor(amplitudes)
        self.noise_std = noise_std
        self.noise_type = noise_type

        if noise_type == "normal":
            self.noise_dist = torch.distributions.Normal(
                torch.tensor(0.0), self.noise_std
            )
        elif noise_type == "laplace":
            self.noise_dist = torch.distributions.Laplace(
                torch.tensor(0.0), self.noise_std / math.sqrt(2)
            )
        else:
            raise NotImplementedError

    def mean_y(self, x: torch.Tensor, latent_indices: torch.Tensor | None = None):
        """
        E[y|x,latent] -> tensor[batch size, # latent options], where x is batched.
        :param x: tensor[batch size, 1]
        :param latent_indices: tensor[batch size] or None
        :return: tensor[batch size, # latent options]
        """
        if latent_indices is None:
            latent_indices = slice(None)
        return (
            self.to_latent_vector(self.y_offsets)[latent_indices][None]
            + self.to_latent_vector(self.amplitudes)[latent_indices][None]
            * (x > self.to_latent_vector(self.x_offsets)[latent_indices][None]).float()
        )

    def sample_xy(self, num_samples=1):
        x = x_sampler(num_samples)
        latent_index = self.sample_latent_index()
        y = self.mean_y(x, latent_index)[:, 0]
        y += self.noise_dist.sample((num_samples,))
        return x, y

    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        """
        if len(y.shape) == 1:
            y = y[:, None]

        mean_y = self.mean_y(x)  # (batch size, #latent options)
        y_logl = self.noise_dist.log_prob(y - mean_y)  # (batch size, #latent options)
        x_logl = x_sampler.logprob(x)  # (batch size, 1)
        return y_logl + x_logl


class Counting(DiscretePrior):
    def __init__(self, weights: torch.Tensor):
        super().__init__(weights)
        self.register_buffer(
            "class_1_prob", torch.linspace(0, 1, len(weights) + 2)[1:][:-1]
        )

    def sample_xy(self, num_samples=1):
        class_1_prob = self.class_1_prob[self.sample_latent_index()].expand(num_samples)
        y = torch.bernoulli(class_1_prob)
        return torch.zeros(num_samples, 1, device=self.device), y

    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        """
        assert (y.bool() == (y == 1)).all()
        log_likelihoods = torch.zeros(y.shape[0], len(self), device=self.device)
        for i, class_prob in enumerate(self.class_1_prob):
            log_likelihoods[:, i] = torch.where(
                y.bool().to(self.device),
                torch.log(class_prob),
                torch.log(1 - class_prob),
            )
        return log_likelihoods

    def probs_y(self, x: torch.Tensor):
        """
        This computes p(y|x,latent) -> tensor[batch size, #latent options, #classes]
        """
        classes_probs = torch.stack([1 - self.class_1_prob, self.class_1_prob], dim=1)
        return classes_probs[None].expand(len(x), -1, -1)


class Sinus(DiscretePrior):
    def __init__(
        self,
        weights: torch.Tensor,
        y_offsets: float | torch.Tensor = 0.0,
        x_offsets: float | torch.Tensor = 0.0,
        amplitudes: float | torch.Tensor = 0.0,
        frequencies: float | torch.Tensor = 1.0,
        noise_std: float = 0.1,
        slopes: float | torch.Tensor = 0.0,
    ):
        super().__init__(weights)
        self.y_offsets = to_tensor(y_offsets)
        self.x_offsets = to_tensor(x_offsets)
        self.amplitudes = to_tensor(amplitudes)
        self.frequencies = to_tensor(frequencies)
        self.noise_std = noise_std
        self.slopes = to_tensor(slopes)

        self.noise_dist = torch.distributions.Normal(torch.tensor(0.0), self.noise_std)

    def mean_y(self, x: torch.Tensor, latent_indices: torch.Tensor | None = None):
        """
        E[y|x,latent] -> tensor[batch size, # latent options], where x is batched.
        :param x: (batch size, 1)
        :param latent_indices: list of ints
        :return: (batch size, # latent options)
        """
        if latent_indices is None:
            latent_indices = slice(None)

        return (
            self.to_latent_vector(self.y_offsets)[latent_indices][None]
            + self.to_latent_vector(self.amplitudes)[latent_indices][None]
            * torch.sin(
                self.to_latent_vector(self.frequencies)[latent_indices][None] * x
                + self.to_latent_vector(self.x_offsets)[latent_indices][None]
                * 2
                * math.pi
            )
            + self.to_latent_vector(self.slopes)[latent_indices][None] * x
        )

    def sample_xy(self, num_samples=1):
        latent_idx = self.sample_latent_index()
        x = x_sampler(num_samples)
        y = self.mean_y(x, latent_idx)[:, 0]
        y += self.noise_dist.sample(y.shape)
        return x, y

    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        """
        if len(y.shape) == 1:
            y = y[:, None]

        mean_y = self.mean_y(x)  # (batch size, #latent options)
        y_logl = self.noise_dist.log_prob(y - mean_y)  # (batch size, #latent options)
        x_logl = x_sampler.logprob(x)  # (batch size, 1)
        return y_logl + x_logl


class Normal(torch.distributions.Normal):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )


class TwoLevelUniform(torch.distributions.Distribution):
    def __init__(
        self,
        inner_range_mins: torch.Tensor,
        inner_range_maxs: torch.Tensor,
        outer_range=(-1, 1),
        prob_inner=0.9,
    ):
        self.inner_range_mins = inner_range_mins
        self.inner_range_maxs = inner_range_maxs
        self.outer_range = outer_range
        self.prob_inner = prob_inner

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        inner_widths = self.inner_range_maxs - self.inner_range_mins
        inner_logl = math.log(self.prob_inner) - torch.log(inner_widths)
        outer_logl = math.log(1 - self.prob_inner) - torch.log(
            self.outer_range[1] - self.outer_range[0] - inner_widths
        )

        inner_mask = (value >= self.inner_range_mins) & (value <= self.inner_range_maxs)
        return torch.where(inner_mask, inner_logl, outer_logl)


class Gaussian2DClassification(DiscretePrior):
    def __init__(
        self,
        weights: torch.Tensor,
        means_per_class: torch.Tensor,
        stds_per_class: torch.Tensor,
        dist_type="normal",
    ):
        """
        This is a prior for a 2D Gaussian binary classification problem.
        Each problem consists of two gaussian distributions, one for each class.
        Classes are balanced, i.e. drawn with equal probability.
        :param weights: tensor[#latent options]
        :param means_per_class: tensor[#latent options, num_classes=2, num_dimensions=2]
        :param stds_per_class: tensor[#latent options, num_classes=2, num_dimensions=2]
        """
        super().__init__(weights)
        assert means_per_class.shape == stds_per_class.shape
        assert means_per_class.shape[0] == len(weights)
        assert means_per_class.shape[1:] == (2, 2)
        self.register_buffer("means_per_class", means_per_class)
        self.register_buffer("stds_per_class", stds_per_class)
        self.dist_type = dist_type

    def sample_xy(self, num_samples=1):
        y = torch.randint(2, (num_samples,))
        latent_index = self.sample_latent_index()[0]
        means = self.means_per_class[latent_index][y]  # (num_samples, 2)
        stds = self.stds_per_class[latent_index][y]  # (num_samples, 2)
        x = torch.randn(num_samples, 2, device=self.device) * stds + means
        return x, y

    def get_x_dist(self):
        if self.dist_type == "normal":
            return torch.distributions.Normal(self.means_per_class, self.stds_per_class)
        elif self.dist_type == "laplace":
            return torch.distributions.Laplace(
                self.means_per_class, self.stds_per_class / math.sqrt(2)
            )
        elif self.dist_type == "two_level_uniform":
            return TwoLevelUniform(
                self.means_per_class - self.stds_per_class,
                self.means_per_class + self.stds_per_class,
            )
        else:
            raise NotImplementedError

    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        :param x: tensor[batch size, 2]
        :param y:
        :return:
        """
        x = x.to(self.device)
        y = y.to(self.device)
        x_dist = self.get_x_dist()
        probs_for_all_classes = x_dist.log_prob(
            x[:, None, None, :].expand(-1, len(self), 2, -1)
        ).sum(-1)  # (batch size, #latent options, 2)
        # the above can get quite large but all we need is actually only the sum without the batch size i think
        log_class_prob = math.log(0.5)
        return probs_for_all_classes[torch.arange(len(x)), :, y] + log_class_prob

    def probs_y(self, x: torch.Tensor):
        x = x.to(self.device)
        x_dist = self.get_x_dist()
        log_probs = x_dist.log_prob(
            x[:, None, None, :].expand(-1, len(self), 2, -1)
        )  # (batch size, #latent options, 2, 2)
        log_probs = log_probs.sum(-1)  # (batch size, #latent options, num_classes=2)
        return torch.softmax(log_probs, dim=2)  # (batch size, #latent options, 2)


class Stroke(DiscretePrior):
    def __init__(
        self,
        resolution=8,
        no_cut_off=False,
        noise_std=0.1,
        noise_type="normal",
    ):
        """
        :param weights: (num_latent_options, )
        """
        self.resolution = resolution
        self.rotations = [(0, 1), (1, 0), (1, 1), (1, -1)]
        self.lengths = list(range(2, self.resolution // 2 + 1))

        self.stroke_types = [(l, r) for l in self.lengths for r in self.rotations]
        self.latents = [
            (i, j)
            for i in range(len(self.stroke_types))
            for j in range(0, len(self.stroke_types))
        ]
        self.no_cut_off = no_cut_off
        super().__init__(torch.ones(len(self.latents)))
        self.noise_type = noise_type
        self.noise_std = noise_std
        if noise_type == "normal":
            self.noise_dist = torch.distributions.Normal(torch.tensor(0.0), noise_std)
        elif noise_type == "uniform":
            pass
        else:
            raise NotImplementedError

        images_per_stroke = []

        def compute_index_offsets(length, right, up):
            i_indexes = torch.arange(length) * right
            j_indexes = torch.arange(length) * up

            return i_indexes, j_indexes

        for length, (right, up) in self.stroke_types:
            i_indexes, j_indexes = compute_index_offsets(length, right, up)
            images_of_this_stroke = []
            for i in range(self.resolution):
                for j in range(self.resolution):
                    i_s = i + i_indexes
                    j_s = j + j_indexes

                    adapted_len = sum(
                        [
                            i >= 0
                            and i < self.resolution
                            and j >= 0
                            and j < self.resolution
                            for i, j in zip(i_s, j_s)
                        ]
                    )

                    if self.no_cut_off and adapted_len < length:
                        continue

                    images_of_this_stroke.append(
                        torch.zeros(self.resolution, self.resolution)
                    )
                    images_of_this_stroke[-1][i_s[:adapted_len], j_s[:adapted_len]] = 1
            images_per_stroke.append(torch.stack(images_of_this_stroke))
        self.images_per_stroke = images_per_stroke

    def sample_xy(
        self,
        num_samples=1,
        sample_among_first_n_variants=None,
        sample_half_half=False,
    ):
        """
        Sample from p(x, y) num_samples times, from the latent options
        Return x: tensor, y: tensor
        """
        latent_idx = self.sample_latent_index()[0]
        stroke_inds = self.latents[latent_idx]

        x = torch.zeros(num_samples, self.resolution, self.resolution)
        ys = (torch.rand(num_samples) > 0.5).int()
        if sample_half_half:
            ys[: num_samples // 2] = 0
            ys[num_samples // 2 :] = 1

        for bi, y in enumerate(ys):
            images_of_this_stroke = self.images_per_stroke[stroke_inds[y]]
            if sample_among_first_n_variants is not None:
                variant_index = torch.randint(sample_among_first_n_variants, tuple())
            else:
                variant_index = torch.randint(len(images_of_this_stroke), tuple())

            x[bi] = images_of_this_stroke[variant_index]

        if self.noise_type == "normal":
            x += self.noise_dist.sample(x.shape)
        elif self.noise_type == "uniform":
            x[x == 0.0] = torch.rand(x[x == 0.0].shape)
            x[x == 1.0] = 1.0 - torch.rand(x[x == 1.0].shape) * self.noise_std

        return x.view(num_samples, self.resolution**2), ys

    def calculate_log_likelihoods_for_images(self, x: torch.Tensor):
        x = x.to(self.device)
        if self.noise_type == "normal":
            log_likelihoods = [
                self.noise_dist.log_prob(x[:, None] - images_for_stroke[None, :])
                .sum(-1)
                .sum(-1)
                for images_for_stroke in self.images_per_stroke
            ]
        elif self.noise_type == "uniform":
            log_likelihoods_one = (
                (x >= (1 - self.noise_std)).float() / self.noise_std
            ).log()
            log_likelihoods_zero = torch.zeros_like(log_likelihoods_one)
            assert all(
                ((images_for_stroke == 0.0) | (images_for_stroke == 1.0)).all()
                for images_for_stroke in self.images_per_stroke
            )
            log_likelihoods = [
                torch.where(
                    images_for_stroke[None].bool(),
                    log_likelihoods_one[:, None],
                    log_likelihoods_zero[:, None],
                )
                .sum(-1)
                .sum(-1)
                for images_for_stroke in self.images_per_stroke
            ]
        log_likelihoods = [
            torch.logsumexp(log_likelihood, dim=-1)
            - torch.log(torch.tensor(len(images_for_stroke)))
            for log_likelihood, images_for_stroke in zip(
                log_likelihoods, self.images_per_stroke
            )
        ]
        log_likelihoods = torch.stack(
            log_likelihoods, dim=1
        )  # (batch size, #stroke types)
        return log_likelihoods

    def xy_logprob(self, x: torch.Tensor, y: torch.Tensor):
        """
        This computes (across all latens) log p(x, y|latent) -> torch.Tensor[batch size, #latent options], where x and y are batched
        """

        log_likelihoods = self.calculate_log_likelihoods_for_images(
            x
        )  # (batch size, #stroke types)
        print(log_likelihoods.shape)

        # log_probs = torch.zeros(x.shape[0], len(self.latents), device=self.device)
        # for i, (s1, s2) in enumerate(self.latents):
        #     log_probs[:, i] = torch.where(y.bool().to(self.device), log_likelihoods[:, s2], log_likelihoods[:, s1])

        s1_indices = torch.tensor([s1 for s1, s2 in self.latents], device=self.device)
        s2_indices = torch.tensor([s2 for s1, s2 in self.latents], device=self.device)
        log_probs = torch.where(
            y.bool().to(self.device)[:, None],
            log_likelihoods[:, s2_indices],
            log_likelihoods[:, s1_indices],
        )
        print(log_probs.shape)

        return log_probs

    def probs_y(self, x: torch.Tensor):
        """
        This computes p(y|x,latent) -> tensor[batch size, #latent options, #classes]
        This function is only used for classification.
        """
        log_likelihoods = self.calculate_log_likelihoods_for_images(x)

        # log_probs = torch.zeros(x.shape[0], len(self), 2)
        # for i, (s1, s2) in enumerate(self.latents):
        #     log_probs[:, i] = log_likelihoods[:, [s1, s2]]
        indices = torch.tensor(self.latents, device=self.device)
        log_probs = log_likelihoods[:, indices]
        return torch.softmax(log_probs, dim=2)

    def averaged_probs_y(self, x: torch.Tensor, weights: torch.Tensor):
        """
        This computes p(y|x,latent) @ latent weights -> tensor[batch size, #classes]
        This function is only used for classification.
        """
        indices = torch.tensor(
            self.latents, device=self.device
        )  # (num_latent_options, 2)
        log_likelihoods = self.calculate_log_likelihoods_for_images(
            x
        )  # (batch_size, stroke_types (sqrt(num_latent_options))
        # log_probs = log_likelihoods[:, indices] # (batch_size, num_latent_options, 2)
        # return torch.softmax(log_probs, dim=2) @ weights
        out_probs = torch.zeros(len(x), 2, device=self.device)
        batch_size = 100  # Specify the batch size
        num_latents = len(self.latents)

        for i in range(0, num_latents, batch_size):
            batch_end = min(i + batch_size, num_latents)

            # Compute probabilities for this batch of latents
            probs_for_batch = torch.softmax(
                log_likelihoods[:, indices[i:batch_end]], dim=2
            )

            # Multiply by weights and sum for this batch
            out_probs += torch.einsum(
                "blc,l->bc", probs_for_batch, weights[i:batch_end]
            )

        # Normalize the output probabilities
        out_probs /= weights.sum()
        return out_probs


# todo actually include x in the conditioning in xy_logprob


class StrokeNotCompletelyTranslationInvariant(Stroke):
    def __init__(
        self,
        resolution=8,
        noise_std=0.1,
        noise_type="normal",
        max_translations=(3, 3),
        lengths=None,
    ):
        """
        :param weights: (num_latent_options, )
        """

        self.noise_type = noise_type
        self.noise_std = noise_std
        if noise_type == "normal":
            self.noise_dist = torch.distributions.Normal(torch.tensor(0.0), noise_std)
        elif noise_type == "uniform":
            pass
        else:
            raise NotImplementedError

        self.resolution = resolution
        self.rotations = [(0, 1), (1, 0), (1, 1), (1, -1)]
        if lengths is None:
            self.lengths = list(range(2, self.resolution // 2 + 1))
        else:
            self.lengths = lengths

        def compute_index_offsets(length, right, up, i_offset, j_offset):
            i_indexes = torch.arange(length) * right + i_offset
            j_indexes = torch.arange(length) * up + j_offset

            return i_indexes, j_indexes

        unfiltered_stroke_types = [
            (l, r, i_offset, j_offset)
            for l in self.lengths
            for r in self.rotations
            for i_offset in range(self.resolution)
            for j_offset in range(self.resolution)
        ]

        self.stroke_types = []

        for length, (right, up), i_offset, j_offset in unfiltered_stroke_types:
            i_indexes, j_indexes = compute_index_offsets(
                length, right, up, i_offset, j_offset
            )

            if (
                (max(i_indexes + max_translations[0]) >= self.resolution)
                or (max(j_indexes + max_translations[1]) >= self.resolution)
                or (min(i_indexes - max_translations[0]) < 0)
                or (min(j_indexes - max_translations[1]) < 0)
            ):
                continue
            self.stroke_types.append((length, (right, up), i_offset, j_offset))

        self.latents = [
            (i, j)
            for i in range(len(self.stroke_types))
            for j in range(0, len(self.stroke_types))
        ]

        DiscretePrior.__init__(self, torch.ones(len(self.latents)))

        images_per_stroke = []

        for length, (right, up), i_offset, j_offset in self.stroke_types:
            i_indexes, j_indexes = compute_index_offsets(
                length, right, up, i_offset, j_offset
            )
            images_of_this_stroke = []
            for i_translation in range(-max_translations[0], max_translations[0] + 1):
                for j_translation in range(
                    -max_translations[1], max_translations[1] + 1
                ):
                    i_s = i_indexes + i_translation
                    j_s = j_indexes + j_translation

                    images_of_this_stroke.append(
                        torch.zeros(self.resolution, self.resolution)
                    )
                    images_of_this_stroke[-1][i_s, j_s] = 1
            images_per_stroke.append(torch.stack(images_of_this_stroke))
        self.register_buffer("images_per_stroke", torch.stack(images_per_stroke))


def shift_pixels(image, shift_x, shift_y):
    height, width = image.shape

    # Pad the image
    pad_left = max(-shift_x, 0)
    pad_right = max(shift_x, 0)
    pad_top = max(-shift_y, 0)
    pad_bottom = max(shift_y, 0)

    padded = F.pad(
        image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    # Calculate the starting indices for slicing
    start_x = pad_left + shift_x
    start_y = pad_top + shift_y

    # Slice the padded image to get the shifted result
    result = padded[start_y : start_y + height, start_x : start_x + width]

    return result


@torch.no_grad()
def get_batch_random_pixels(
    batch_size,
    seq_len,
    num_features,
    hyperparameters=None,
    device="cpu",
    normalize=False,
    **kwargs,
):
    if hyperparameters is None:
        hyperparameters = {"max_translation": 0, "draw_prob": 0.5, "noise_std": 0.1}
    res = int(math.sqrt(num_features))
    assert res**2 == num_features, "num_features must be a perfect square"

    max_translation = hyperparameters["max_translation"]
    if max_translation is None:
        max_translation = res

    batch = []
    for _ in range(batch_size):
        prototypes = torch.rand(2, res, res) < hyperparameters["draw_prob"]
        y = torch.randint(2, (seq_len,))
        x = torch.zeros(seq_len, res, res)
        # Generate random translations for all sequence elements at once
        translations_i = torch.randint(
            -max_translation, max_translation + 1, (seq_len,)
        )
        translations_j = torch.randint(
            -max_translation, max_translation + 1, (seq_len,)
        )

        # Create a tensor of indices for batch processing
        indices = torch.arange(seq_len)

        # Create translated versions of the prototypes for all sequence elements
        x = torch.stack(
            [
                # torch.roll(prototypes[y[i]], shifts=(translations_i[i].item(), translations_j[i].item()), dims=(0, 1))
                shift_pixels(
                    prototypes[y[i]],
                    translations_i[i].item(),
                    translations_j[i].item(),
                )
                for i in indices
            ]
        ).float()

        if hyperparameters["noise_std"] > 0:
            x += torch.randn_like(x) * hyperparameters["noise_std"]

        if normalize:
            x = (x - x.mean()) / x.std()

        batch.append((x, y))

    return Batch(
        x=torch.stack([x for x, _ in batch], 1)
        .to(device)
        .view(seq_len, batch_size, num_features),
        y=torch.stack([y for _, y in batch], 1).clone().to(device).float(),
        target_y=torch.stack([y for _, y in batch], 1).clone().to(device).float(),
    )


class DiscreteBayes(torch.nn.Module):
    def __init__(
        self,
        priors: list[DiscretePrior],
        meta_weights: torch.Tensor | None = None,
    ):
        """

        :param priors: a list of types of latent options
        :param meta_weights: a list of weights for each type of latent options
        """
        super().__init__()
        self.priors = torch.nn.ModuleList(priors)
        if meta_weights is None:
            meta_weights = torch.ones(len(priors))
        self.register_buffer("meta_weights", meta_weights)

    def sample_latent_options_type(self):
        """
        Sample from p(latent)
        """
        return self.priors[torch.multinomial(self.meta_weights, 1).item()]

    @torch.no_grad()
    def get_batch(
        self,
        batch_size,
        seq_len,
        num_features,
        hyperparameters=None,
        device="cpu",
        normalize=False,
        **kwargs,
    ):
        if hyperparameters is None:
            hyperparameters = {}
        batch = []
        for _ in range(batch_size):
            latent_options = self.sample_latent_options_type()
            x, y = latent_options.sample_xy(seq_len, **(hyperparameters or {}))
            if len(y.shape) == 1:
                y = y[:, None]
            batch.append((x, y))
        assert num_features == x.shape[1]
        x = torch.stack([x for x, _ in batch], 1).to(device)
        if normalize:
            x = (x - x.mean()) / x.std()
        return Batch(
            x=x,
            y=torch.stack([y for _, y in batch], 1).clone().to(device).float(),
            target_y=torch.stack([y for _, y in batch], 1).clone().to(device).float(),
        )

    @torch.no_grad()
    def compute_log_likelihood(self, x: torch.Tensor, y: torch.Tensor):
        """
        Computes log p(D|latent) for all latents or p(D,x|latent), if x is one longer than y.
        """
        # TODO: actually condition on the query x too
        all_log_likelihoods = []

        for priors in self.priors:
            all_log_likelihoods.append(priors.xy_logprob(x, y).sum(0))

        return all_log_likelihoods

    @torch.no_grad()
    def compute_log_posterior(
        self, x: torch.Tensor, y: torch.Tensor
    ):  # only works on a single dataset atm
        """
        Computes p(latent|D) for all latents, as well as the unnormalized posterior (p(D,latent)) and the normalizing term (p(D)).
        """
        all_log_likelihoods = self.compute_log_likelihood(x, y)  # list of vectors

        # Define prior probabilities p(prior) and p(latent|prior)
        meta_priors = self.meta_weights / sum(self.meta_weights)
        log_priors = [prior.prior_logprobs for prior in self.priors]

        print(
            all_log_likelihoods[0].device,
            log_priors[0].device,
            meta_priors.device,
        )

        # Compute unnormailized posterior probabilities, p(D|latent)p(latent)
        unnormalized_log_posterior = [
            log_likelihood + torch.log(meta_prior) + log_prior
            for log_likelihood, meta_prior, log_prior in zip(
                all_log_likelihoods, meta_priors, log_priors
            )
        ]

        # Compute posterior
        normalizing_term = torch.logsumexp(
            torch.cat(unnormalized_log_posterior), dim=0
        )  # scalar
        log_posterior = [
            log_prob - normalizing_term for log_prob in unnormalized_log_posterior
        ]

        return log_posterior, unnormalized_log_posterior, normalizing_term

    @torch.no_grad()
    def posterior_predictive_mean(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_query: torch.Tensor,
        verbose_output=False,
    ):
        """
        Computes the posterior predictive mean for a for a regression prior for a single dataset.
        """
        log_posterior, unnormalized_log_posterior, normalizing_term = (
            self.compute_log_posterior(x, y)
        )  # list of vectors

        total_p = sum(lp.double().exp().sum() for lp in log_posterior).item()
        assert (
            abs(total_p - 1) < 1e-1
        ), f"Posterior probabilities do not sum to 1: {total_p}"

        posterior_predictive_mean = 0.0

        for log_posterior_for_this_prior, prior in zip(log_posterior, self.priors):
            try:
                posterior_predictive_mean += (
                    torch.exp(log_posterior_for_this_prior) @ prior.mean_y(x_query).T
                )
            except NotImplementedError:
                if hasattr(prior, "averaged_probs_y"):
                    posterior_predictive_mean += prior.averaged_probs_y(
                        x_query, torch.exp(log_posterior_for_this_prior)
                    )
                else:
                    posterior_predictive_mean += torch.einsum(
                        "l,tlc->tc",
                        torch.exp(log_posterior_for_this_prior),
                        prior.probs_y(x_query),
                    )

        if verbose_output:
            return (
                posterior_predictive_mean,
                log_posterior,
                unnormalized_log_posterior,
                normalizing_term,
            )
        else:
            return posterior_predictive_mean
