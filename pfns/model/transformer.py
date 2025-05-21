from __future__ import annotations

import random
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

import einops
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pfns.model.encoders import (
    SequentialEncoder,
    get_linear_x_encoder,
    get_linear_y_encoder,
)
from pfns.model.layer import PerFeatureLayer

DEFAULT_EMSIZE = 128


class PerFeatureTransformer(nn.Module):
    """A Transformer model processes a token per feature and sample.

    This model extends the standard Transformer architecture to operate on a
    per-feature basis.
    It allows for processing each feature separately while still leveraging the
    power of self-attention.

    The model consists of an encoder, decoder, and optional components such
    as a feature positional embedding and a separate decoder for each feature.
    """

    # TODO: Feel like this could be simplified a lot from this part downwards
    def __init__(  # noqa: C901, D417, PLR0913
        self,
        *,
        encoder: nn.Module | None = None,
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        y_encoder: nn.Module | None = None,
        decoder_dict: (
            dict[str,
                 tuple[
                     Callable[[int, int, int], nn.Module] | None,
                     int
                     ]
                ]
            | None
        ) = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec",
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        zero_init: bool = True,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        style_encoder: nn.Module | None = None,
        y_style_encoder: nn.Module | None = None,
        attention_between_features: bool = True,
        **layer_kwargs: Any,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            encoder:
                Pass a nn.Module that takes in a batch of sequences of inputs and
                returns something of the shape (seq_len, batch_size, ninp)
            ninp: Input dimension, also called the embedding dimension
            nhead: Number of attention heads
            nhid: Hidden dimension in the MLP layers
            nlayers:
                Number of layers, each consisting of a multi-head attention layer and
                an MLP layer
            y_encoder:
                A nn.Module that takes in a batch of sequences of outputs and
                returns something of the shape (seq_len, batch_size, ninp)
            decoder_dict: A mapping from output keys to a tuple of a decoder model and the number of output neurons.
                The number of output neurons for 10-way classification is 10 for example, and for regression with a bar distribution
                with 1000 buckets, it is 1000.
                If the decoder model is None, an MLP decoder is used, if one wants to specify it, it has the signature:
                    decoder_model(
                        ninp,
                        nhid,
                        decoder_n_out,
                    )
            activation: An activation function, "gelu" or "relu"
            recompute_layer:
                If True, the transformer layers will be recomputed on each
                forward pass in training. This is useful to save memory.
            min_num_layers_layer_dropout:
                If this is set, it enables to drop the last
                layers randomly during training up to this number.
            features_per_group:
                If > 1, the features will be grouped into groups of this
                size and the attention is across groups.
            feature_positional_embedding:
                There is a risk that our models confuse
                features with each other. This positional embedding is added to the
                features to help the model distinguish them.
                We recommend setting this to "subspace".
            zero_init:
                If True, the last sublayer of each attention and MLP layer will
                be initialized with zeros.
                Thus, the layers will start out as identity functions.
            seed: The seed to use for the random number generator.
            precomputed_kv: Experimental
            style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1)
                or one style vector per feature (batch_size, num_features, -1) and returns a style embedding of the shape (batch_size, ninp)
            y_style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1) and returns a style embedding of the shape (batch_size, ninp)
            attention_between_features: If True, apply attention between feature groups. If False, use the old PFN architecture, see https://github.com/automl/TransformersCanDoBayesianInference
            layer_kwargs: Keyword arguments passed to the `PerFeatureEncoderLayer`.
                Possible arguments include:
                - `layer_norm_eps`: Epsilon for layer normalization.
                - `pre_norm`: Apply layer norm before attention/MLP.
                - `recompute_attn`: Recompute attention during backpropagation.
                - `second_mlp`: Include a second MLP in the layer.
                - `layer_norm_with_elementwise_affine`: Use elementwise affine in layer norm.
                - `save_peak_mem_factor`: Factor to save peak memory (post-norm only).
                - `multiquery_item_attention`: Use multiquery attention for all items.
                - `multiquery_item_attention_for_test_set`: Use multiquery attention for the test set.
                - `attention_init_gain`: Gain for initializing attention parameters.
                - `d_k`: Dimensionality of query and key vectors.
                - `d_v`: Dimensionality of value vectors.
        """
        if decoder_dict is None:
            decoder_dict = {"standard": (None, 1)}

        super().__init__()

        if encoder is None:
            print("Using linear x encoder, as no encoder was provided.")
            encoder = get_linear_x_encoder(ninp, features_per_group)

        if y_encoder is None:
            print("Using linear y encoder, as no y_encoder was provided.")
            y_encoder = get_linear_y_encoder(ninp)

        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None
        self.attention_between_features = attention_between_features

        layer_creator = lambda: PerFeatureLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            attention_between_features=attention_between_features,
            **layer_kwargs,
        )

        nlayers_encoder = nlayers

        self.transformer_layers = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )


        initialized_decoder_dict = {}
        for decoder_key in decoder_dict:
            decoder_model, decoder_n_out = decoder_dict[decoder_key]
            if decoder_model is None:
                initialized_decoder_dict[decoder_key] = nn.Sequential(
                    nn.Linear(ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, decoder_n_out),
                )
            else:
                initialized_decoder_dict[decoder_key] = decoder_model(
                    ninp,
                    nhid,
                    decoder_n_out,
                )
        self.decoder_dict = nn.ModuleDict(initialized_decoder_dict)

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)  # noqa: S311

        self.style_encoder = style_encoder
        if y_style_encoder is not None:
            assert attention_between_features, "Attention between features must be True when using a y_style_encoder, otherwise only use a style_encoder."
        self.y_style_encoder = y_style_encoder
        

    def forward(self, x: torch.Tensor | None, y: torch.Tensor | None, test_x: torch.Tensor | None = None, style: torch.Tensor | None = None, y_style: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:  # noqa: D417
        """
        x can either contain both the train and test part, or the test part can be passed as test_x.

        Args:
            x: (seq_len_train | seq_len_train + seq_len_test, batch_size, num_features) The input data for the training set, or both the train and test part if test_x is None.
                When predicting from cached trainset representations, x can be None or contain the test set.
            y: (seq_len_train | seq_len_train + seq_len_test, batch_size, num_targets) The target data for the training set, where num targets is typically 1. In which case the last dimension can be omitted.
                If y is None, we perform predictions for the test set using cached trainset representations.
            test_x: (seq_len_test, batch_size, num_features) The input data for the test set.
                When predicting from cached trainset representations, test_x can be None (using x instead) or contain the test set.
            style: (batch_size, style_dim) or (batch_size, num_features, style_dim) The style vector
            y_style: (batch_size, style_dim) The style vector for the y data
            **kwargs: Keyword arguments passed to the `_forward` method:
                - `only_return_standard_out`: Whether to only return the standard output.
                - `categorical_inds`: The indices of categorical features. A single list of indices for the whole batch:
                    these are shared between the datasets within a batch.
                - `half_layers`: Whether to use the first half of the layers.
        """

        single_eval_pos = len(y) if y is not None else None

        if self.cache_trainset_representation and y is None:
            assert (test_x is None) != (x is None), "Provide the test inputs only via test_x or x, not both, when cache_trainset_representation is True"
            if test_x is not None:
                x = test_x
        else:
            assert x is not None, "x must be provided when not predicting from cached trainset representations"
            assert y is not None, "y must be provided when not predicting from cached trainset representations"

            if test_x is not None:
                assert single_eval_pos == len(x)
                x = torch.cat((x, test_x), dim=0)
        
        return self._forward(x, y, single_eval_pos=single_eval_pos, style=style, y_style=y_style, **kwargs)

    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,
        y: torch.Tensor | dict | None,
        *,
        single_eval_pos: int | None = None,
        only_return_standard_out: bool = True,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model.

        Args:
            x: The input data. Shape: `(seq_len, batch_size, num_features)`. This can also be a dictionary of tensors,
                the default being a tensor with key "main", but can be others depending on the encoder.
            y: The target data. Shape: `(seq_len, batch_size, num_targets)` where num targets is typically 1. 
                In which case the last dimension can be omitted. Can be None if we perform predictions
                for the test set using cached trainset representations. This can also be a dictionary of tensors,
                the default being a tensor with key "main", but can be others depending on the y_encoder.
            single_eval_pos:
                The position to evaluate at. If `None`, evaluate at all positions using the cached trainset representations.
            only_return_standard_out: Whether to only return the standard output.
            style: (batch_size, style_dim) or (batch_size, num_features, style_dim) The style vector
            y_style: (batch_size, style_dim) The style vector for the y data
            categorical_inds: The indices of categorical features. A single list of indices for the whole batch:
                these are shared between the datasets within a batch.
            half_layers: Whether to use the first half of the layers.

        Returns:
            A dictionary of output tensors.
        """
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert y is None
        else:
            assert y is not None
            assert single_eval_pos

        single_eval_pos_ = single_eval_pos or 0
        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_features = x["main"].shape

        if y is None:
            # TODO: check dtype.
            y = torch.zeros(
                0,
                batch_size,
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}
        
        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )
                if style is not None and style.ndim == 3:
                    style = torch.cat(
                        (
                            style,
                            torch.zeros(batch_size, missing_to_next, style.shape[2], device=style.device, dtype=style.dtype),
                        ),
                        dim=-2,
                    )

        # Splits up the input into subgroups
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  # s b f -> b s #groups #features_per_group

        if style is not None:
            # Split up the style, if it is provided
            # and has a feature dimension
            if style.ndim == 3:
                batched_style = einops.rearrange(style, "b (f n) s -> (b f) n s", n=self.features_per_group)  # s represents the style dimension
            else:
                assert style.ndim == 2
                batched_style = einops.repeat(style, "b s -> (b f) s", f=num_features)  # b s -> (b f) s
        
        # We have to re-work categoricals based on the subgroup they fall into.
        categorical_inds_to_use: list[list[int]] | None = None
        if categorical_inds is not None:
            new_categorical_inds = []
            n_subgroups = x["main"].shape[2]

            for subgroup in range(n_subgroups):
                subgroup_lower = subgroup * self.features_per_group
                subgroup_upper = (subgroup + 1) * self.features_per_group
                subgroup_indices = [
                    i - subgroup_lower
                    for i in categorical_inds
                    if subgroup_lower <= i < subgroup_upper
                ]
                new_categorical_inds.append(subgroup_indices)

            categorical_inds_to_use = new_categorical_inds

        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos_
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos_, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos_:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos_:] = torch.nan

        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y
        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )

        extra_encoders_args = {}
        if categorical_inds_to_use is not None and isinstance(
            self.encoder,
            SequentialEncoder,
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds_to_use

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,
            embedded_y,
            num_features=num_features,
            seq_len=seq_len,
            cache_embeddings=(
                self.cache_trainset_representation and single_eval_pos is not None
            ),
            use_cached_embeddings=(
                self.cache_trainset_representation and single_eval_pos is None
            ),
        )

        if self.attention_between_features:
            # b s f e + b s 1 e -> b s f+1 e
            embedded_input = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        else:
            # add them together in this case, like for the original PFNs
            # b s 1 e + b s 1 e -> b s 1 e
            embedded_input = embedded_x + embedded_y.unsqueeze(2)

        if style is not None:
            embedded_style = self.style_encoder(batched_style) # (batch num_groups) style_dim | (batch num_groups) num_features style_dim -> (batch num_groups) emsize
            embedded_style = einops.rearrange(embedded_style, "(b f) e -> b 1 f e", b=batch_size)  # (batch num_groups) emsize -> batch 1 num_groups emsize
        else:
            embedded_style = None

        if y_style is not None:
            embedded_y_style = self.y_style_encoder(y_style) # batch style_dim -> batch emsize
            embedded_y_style = einops.rearrange(embedded_y_style, "b e -> b 1 1 e")  # batch emsize -> batch 1 1 emsize
        else:
            embedded_y_style = None
        

        if embedded_style is not None or embedded_y_style is not None:
            if embedded_style is None:
                embedded_style = torch.zeros(batch_size, 1, num_features, embedded_input.shape[3], device=embedded_input.device, dtype=embedded_input.dtype)
            
            if embedded_y_style is None:
                embedded_y_style = torch.zeros(batch_size, 1, 1, embedded_input.shape[3], device=embedded_input.device, dtype=embedded_input.dtype)
            
            full_embedded_style = torch.cat((embedded_style, embedded_y_style), dim=2)
            
            embedded_input = torch.cat((full_embedded_style, embedded_input), dim=1)
            single_eval_pos_ += 1  # we added a style embedding

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_layers(
            embedded_input,
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s f+1 e -> b s f+1 e

        del embedded_input

        # out: s b e
        test_encoder_out = encoder_out[:, single_eval_pos_:, -1].transpose(0, 1)

        if only_return_standard_out:
            assert self.decoder_dict is not None
            output_decoded = self.decoder_dict["standard"](test_encoder_out)
        else:
            output_decoded = (
                {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
                if self.decoder_dict is not None
                else {}
            )

            # out: s b e
            train_encoder_out = encoder_out[:, :single_eval_pos_, -1].transpose(0, 1)
            output_decoded["train_embeddings"] = train_encoder_out
            output_decoded["test_embeddings"] = test_encoder_out

        return output_decoded

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        num_features: int,
        seq_len: int,
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_embeddings and self.cached_embeddings is not None:
            x += self.cached_embeddings[None, None]
            return x, y

        with isolate_torch_rng(self.seed, device=x.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (x.shape[2], x.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (x.shape[2], x.shape[3]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    * 2
                    - 1
                )
                x += embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (x.shape[2],),
                    )
                ]
                x += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=x.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                x += embs[None, None]
            elif self.feature_positional_embedding is None:
                embs = None
            else:
                raise ValueError(f"Unknown {self.feature_positional_embedding=}")

        self.cached_embeddings = None
        if cache_embeddings and embs is not None:
            self.cached_embeddings = embs

        return x, y

    def empty_trainset_representation_cache(self) -> None:
        for layer in self.transformer_layers.layers:
            layer.empty_trainset_representation_cache()
    
    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers.

        This factor controls how much memory is saved during the forward pass
        in inference mode.

        Setting this factor > 1 will cause the model to save more memory during
        the forward pass in inference mode.

        A value of 8 is good for a 4x larger width in the fully-connected layers.
        and yields a situation were we need around
        `2*num_features*num_items*emsize*2` bytes of memory

        for a forward pass (using mixed precision).

        WARNING: It should only be used with post-norm.

        Args:
            factor: The save_peak_mem_factor to set. Recommended value is 8.
        """
        for layer in self.transformer_layers.layers:
            assert hasattr(
                layer,
                "save_peak_mem_factor",
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor  # type: ignore


### Utility functions

class LayerStack(nn.Module):
    """Same as nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times, which is passed as creater function.
    
    This is used as transformer encoder and decoder.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        """
        Args:
            layer_creator: A function that returns the layer as a nn.Module.
            num_layers: The number of layers to stack.
            recompute_each_layer: If True, the layers will be recomputed on each
                forward pass in training. This is useful to save memory.
            min_num_layers_layer_dropout: If this is set, it enables to drop the last
                layers randomly during training up to this number.
        """
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


### Utility functions

@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    """
    Use the specified seed within the context manager (`with isolate_torch_rng(...)`)
    and return to the original state after the context manager exits.
    """
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)