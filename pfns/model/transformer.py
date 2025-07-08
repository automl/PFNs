from __future__ import annotations

import random
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

import einops
import torch

from pfns.model.encoders import (
    get_linear_x_encoder,
    get_linear_y_encoder,
    SequentialEncoder,
)
from pfns.model.layer import PerFeatureLayer
from torch import nn
from torch.utils.checkpoint import checkpoint

DEFAULT_EMSIZE = 128


class TableTransformer(nn.Module):
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
            dict[str, tuple[Callable[[int, int, int], nn.Module] | None, int]] | None
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
        batch_first: bool = True,
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
            seed: The seed to use for the random embeddings that identify features.
            precomputed_kv: Experimental
            style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1)
                or one style vector per feature (batch_size, num_features, -1) and returns a style embedding of the shape (batch_size, ninp)
            y_style_encoder: A nn.Module that per dataset takes in a single style vector (batch_size, -1) and returns a style embedding of the shape (batch_size, ninp)
            attention_between_features: If True, apply attention between feature groups. If False, use the old PFN architecture, see https://github.com/automl/TransformersCanDoBayesianInference
            batch_first: If True, then the input and output tensors are provided
                as (batch, seq, feature). Default is True. If False,
                (seq, batch, feature).
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
        self.batch_first = batch_first

        def layer_creator():
            return PerFeatureLayer(
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

    def forward(
        self,
        x: torch.Tensor | None,
        y: torch.Tensor | None,
        test_x: torch.Tensor | None = None,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        only_return_standard_out: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:  # noqa: D417
        """
        x can either contain both the train and test part, or the test part can be passed as test_x.

        Args:
            x: The input data for the training set, or both the train and test part if test_x is None.
                Shape: (batch_size, seq_len_train | seq_len_train + seq_len_test, num_features) if batch_first=True,
                else (seq_len_train | seq_len_train + seq_len_test, batch_size, num_features).
                When predicting from cached trainset representations, x can be None or contain the test set.
            y: The target data for the training set, where num targets is typically 1. In which case the last dimension can be omitted.
                Shape: (batch_size, seq_len_train | seq_len_train + seq_len_test, num_targets) if batch_first=True,
                else (seq_len_train | seq_len_train + seq_len_test, batch_size, num_targets).
                If y is None, we perform predictions for the test set using cached trainset representations.
            test_x: The input data for the test set.
                Shape: (batch_size, seq_len_test, num_features) if batch_first=True,
                else (seq_len_test, batch_size, num_features).
                When predicting from cached trainset representations, test_x can be None (using x instead) or contain the test set.
            style: (batch_size, style_dim) or (batch_size, num_features, style_dim) The style vector. Assumed batch-first.
            y_style: (batch_size, style_dim) The style vector for the y data. Assumed batch-first.
            only_return_standard_out: If True, only the standard output is returned.
            **kwargs: Keyword arguments passed to the `_forward` method:
                - `categorical_inds`: The indices of categorical features. A single list of indices for the whole batch:
                    these are shared between the datasets within a batch.
                - `half_layers`: Whether to use the first half of the layers.
        """

        # Prepare batch-first versions of x, y, test_x for _forward
        # and clone all to be sure not to change outside data
        x_bf = x.clone() if x is not None else None
        y_bf = y.clone() if y is not None else None
        test_x_bf = test_x.clone() if test_x is not None else None

        if not self.batch_first:
            if x_bf is not None:
                x_bf = x_bf.transpose(0, 1)
            if y_bf is not None:
                # Ensure y_bf is a tensor before transposing. _forward will handle dict conversion.
                if y_bf.numel() > 0:
                    y_bf = y_bf.transpose(0, 1)
            if test_x_bf is not None:
                test_x_bf = test_x_bf.transpose(0, 1)

        # Now x_bf, y_bf, test_x_bf are batch-first (or None)

        # Determine single_eval_pos based on the original y shape
        if y_bf is not None:
            single_eval_pos = y_bf.shape[1]
        else:
            single_eval_pos = None

        # Handle cache_trainset_representation and combining x, test_x
        if self.cache_trainset_representation and y is None:
            assert (
                (test_x is None) != (x is None)
            ), "Provide the test inputs only via test_x or x, not both, when cache_trainset_representation is True"
            if test_x is not None:
                x_bf = test_x_bf
        else:
            assert (
                x_bf is not None
            ), "x must be provided when not predicting from cached trainset representations"
            assert (
                y is not None
            ), "y must be provided when not predicting from cached trainset representations"

            if test_x_bf is not None:
                # x_bf and test_x_bf are batch-first. Concatenate along sequence dim (1).
                assert (
                    x_bf.shape[1] == single_eval_pos
                ), f"Batch-first x sequence length {x_bf.shape[1]} must match single_eval_pos {single_eval_pos} for concatenation"
                x_bf = torch.cat((x_bf, test_x_bf), dim=1)

        # Call _forward with batch-first tensors
        output_decoded = self._forward(
            x_bf,
            y_bf,
            single_eval_pos=single_eval_pos,  # This is the length of the training part of the sequence
            style=style,  # style is assumed batch-first from input
            y_style=y_style,  # y_style is assumed batch-first from input
            **kwargs,  # contains only_return_standard_out, categorical_inds, half_layers
        )

        # If original input was sequence-first, transpose outputs back
        if not self.batch_first:
            for key, value in output_decoded.items():
                output_decoded[key] = value.transpose(0, 1)
        if only_return_standard_out:
            output_decoded = output_decoded["standard"]

        return output_decoded

    def _forward(  # noqa: PLR0912, C901
        self,
        x: torch.Tensor | dict,  # Expected to be batch-first
        y: torch.Tensor | dict | None,  # Expected to be batch-first
        *,
        single_eval_pos: int
        | None = None,  # Length of the training part of the sequence
        style: torch.Tensor | None = None,  # Assumed batch-first
        y_style: torch.Tensor | None = None,  # Assumed batch-first
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:
        """The core forward pass of the model. Assumes batch-first inputs for x and y."""
        # Assertions and initial setup
        if self.cache_trainset_representation:
            if not single_eval_pos:  # none or 0
                assert (
                    y is None
                ), "_forward expects y=None if single_eval_pos is 0/None and caching"
        else:
            assert (
                y is not None
            ), "_forward expects y if not caching for pure inference or during training"
            assert (
                single_eval_pos is not None
            ), "_forward expects single_eval_pos if not caching for pure inference or during training"

        # single_eval_pos is the length of the training sequence part.
        # If None (e.g. pure inference from cache), treat as 0.
        current_context_len = single_eval_pos or 0

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:  # x is a tensor
            x = {"main": x}
        # x is now a dict of batch-first tensors: x[k] is (batch_size, seq_len, features)

        _batch_size, _seq_len, _num_features_orig_main = x["main"].shape

        if (
            y is None
        ):  # Should only happen if self.cache_trainset_representation and not single_eval_pos
            y_main_ref = x["main"]
            y = {
                "main": torch.zeros(
                    _batch_size,
                    0,
                    device=y_main_ref.device,
                    dtype=y_main_ref.dtype,
                )
            }  # 0 sequence length
        elif isinstance(y, torch.Tensor):  # y is a tensor
            y = {"main": y}
        # y is now a dict of batch-first tensors: y[k] is (batch_size, seq_len_y, targets)

        # Pad features of x to be multiple of features_per_group
        for k in x:
            # x[k] is (batch_size, seq_len, num_features_k)
            num_features_k = x[k].shape[2]
            missing_to_next = (
                self.features_per_group - (num_features_k % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            x[k].shape[0],  # batch_size
                            x[k].shape[1],  # seq_len
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,  # Pad along feature dimension
                )
                if style is not None and style.ndim == 3 and k == "main":
                    style = torch.cat(
                        (
                            style,
                            torch.zeros(
                                style.shape[0],  # batch_size
                                missing_to_next,  # Padding for feature dimension
                                style.shape[2],  # style_dim
                                device=style.device,
                                dtype=style.dtype,
                            ),
                        ),
                        dim=1,  # Pad along style's feature dimension (dim 1)
                    )

        # Splits up the input into subgroups (batch-first)
        # x[k] from (batch_size, seq_len, num_features_padded) to (batch_size, seq_len, num_groups, features_per_group)
        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "b s (f n) -> b s f n",
                n=self.features_per_group,
            )

        num_groups_main = x["main"].shape[2]  # Number of feature groups in x["main"]

        if style is not None:
            if style.ndim == 3:  # (batch_size, num_features_style_padded, style_dim)
                batched_style = einops.rearrange(
                    style,
                    "b (f n) s_dim -> (b f) n s_dim",
                    n=self.features_per_group,
                )
            else:  # style.ndim == 2, (batch_size, style_dim)
                assert style.ndim == 2
                batched_style = einops.repeat(
                    style, "b s_dim -> (b f) s_dim", f=num_groups_main
                )
        else:
            batched_style = None

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
            # y[k] is (batch_size, current_seq_len_y, num_targets_y)
            if y[k].ndim == 2:  # (B,S) or (B,T)
                y[k] = y[k].unsqueeze(-1)  # B S -> B S 1

            # Pad y sequence length if shorter than x's sequence length (_seq_len)
            if y[k].shape[1] < _seq_len:  # _seq_len is full sequence length from x
                # current_context_len is the length of the training part of y
                assert (
                    y[k].shape[1]
                    == current_context_len  # y should only contain train part if shorter
                    or y[k].shape[1] == _seq_len  # Should not happen if already shorter
                ), f"y[{k}] seq len {y[k].shape[1]} vs train_seq_len {current_context_len} vs x_seq_len {_seq_len}"

                # Only pad if y is for training part or not main y (auxiliary targets might be full length)
                if k != "main" or y[k].shape[1] == current_context_len:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],  # batch_size
                                _seq_len - y[k].shape[1],  # seq_len difference
                                y[k].shape[2],  # num_targets_y
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,  # Pad along sequence dimension (dim 1 for batch-first)
                    )
        # Now y[k] is (batch_size, _seq_len, num_targets_y)

        # Making sure no label leakage ever happens for y["main"] (batch-first indexing)
        # current_context_len is the length of the training data part
        if "main" in y and y["main"].shape[1] > current_context_len:
            y["main"][:, current_context_len:] = torch.nan

        # Prepare y for y_encoder (transpose to sequence-first if y_encoder expects it)
        y_for_y_encoder = {}
        for k_enc, v_enc in y.items():
            y_for_y_encoder[k_enc] = v_enc.transpose(0, 1)  # B S T -> S B T

        embedded_y = self.y_encoder(
            y_for_y_encoder,
            single_eval_pos=current_context_len,  # Length of training part for y_encoder
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y, y_for_y_encoder
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
                single_eval_pos=current_context_len,
                cache_trainset_representation=self.cache_trainset_representation,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x,  # (b s num_groups e)
            embedded_y,  # (b s e)
            num_features=_num_features_orig_main,
            seq_len=_seq_len,
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
            assert (
                embedded_x.shape[2] == 1
            ), f"Only 1 feature per group supported for attention_between_features=False, got {embedded_x.shape=}."
            # b s 1 e + b s 1 e -> b s 1 e
            embedded_input = embedded_x + embedded_y.unsqueeze(2)

        if style is not None:
            embedded_style = self.style_encoder(
                batched_style
            )  # (batch num_groups) style_dim | (batch num_groups) num_features style_dim -> (batch num_groups) emsize
            embedded_style = einops.rearrange(
                embedded_style, "(b f) e -> b 1 f e", b=_batch_size
            )  # (batch num_groups) emsize -> batch 1 num_groups emsize
        else:
            embedded_style = None

        if y_style is not None:
            embedded_y_style = self.y_style_encoder(
                y_style
            )  # batch style_dim -> batch emsize
            embedded_y_style = einops.rearrange(
                embedded_y_style, "b e -> b 1 1 e"
            )  # batch emsize -> batch 1 1 emsize
        else:
            embedded_y_style = None

        if embedded_style is not None or embedded_y_style is not None:
            if embedded_style is None:
                embedded_style = torch.zeros(
                    _batch_size,
                    1,  # Style is a single token in sequence dim
                    num_groups_main,
                    embedded_input.shape[3],  # emsize
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )

            if embedded_y_style is None:
                embedded_y_style = torch.zeros(
                    _batch_size,
                    1,
                    1,  # for the y-token
                    embedded_input.shape[3],  # emsize
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )

            full_embedded_style = torch.cat((embedded_style, embedded_y_style), dim=2)

            embedded_input = torch.cat(
                (full_embedded_style, embedded_input),
                dim=1,  # Concatenate along sequence dimension
            )
            current_context_len += 1  # Context length for attention now includes style

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x and y."
                "Check that you do not feed NaNs or use a NaN-handling enocder."
                "Your embedded x and y returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_y).any()=}",
            )
        del embedded_y, embedded_x

        encoder_out = self.transformer_layers(
            embedded_input,  # (b s_effective (num_groups+1_for_y) e)
            single_eval_pos=current_context_len,  # Pass the context length including style
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # b s (num_groups+1_for_y) e -> b s (num_groups+1_for_y) e

        del embedded_input

        # current_context_len now marks the end of the training/style part in the sequence dimension
        # encoder_out is (batch, seq_with_style, num_tokens_incl_y, embed_dim)
        # We want the output for the y-token (last token in the feature/token dimension)
        # for the test sequence part (after current_context_len).

        test_encoder_out = encoder_out[
            :, current_context_len:, -1
        ]  # (batch, seq_test, embed_dim)
        train_encoder_out = encoder_out[
            :, :current_context_len, -1
        ]  # (batch, seq_train_and_style, embed_dim)

        # No transposition needed here as _forward returns batch-first

        output_decoded = (
            {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
            if self.decoder_dict is not None
            else {}
        )

        output_decoded["train_embeddings"] = train_encoder_out
        output_decoded["test_embeddings"] = test_encoder_out  # Already batch-first

        return output_decoded

    def add_embeddings(  # noqa: C901, PLR0912
        self,
        x: torch.Tensor,  # (b s num_groups e)
        y: torch.Tensor,  # (b s e)
        *,
        num_features: int,  # Original number of features (before grouping)
        seq_len: int,  # Sequence length
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_cached_embeddings and self.cached_embeddings is not None:
            x += self.cached_embeddings[None, None]
            return x, y

        with isolate_torch_rng(self.seed, device=x.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (x.shape[2], x.shape[3]),  # (num_groups, emsize)
                    device=x.device,
                    dtype=x.dtype,
                )
                x += embs[None, None]  # Broadcast across batch and seq_len
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (x.shape[2], x.shape[3]),  # (num_groups, emsize)
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
                        (x.shape[2],),  # num_groups indices
                    )
                ]  # (num_groups, emsize)
                x += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                # x.shape[2] is num_groups, x.shape[3] is emsize
                # Generate (num_groups, emsize // 4) random vectors
                rand_vecs_for_subspace = torch.randn(
                    (x.shape[2], x.shape[3] // 4),
                    device=x.device,
                    dtype=x.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(
                    rand_vecs_for_subspace
                )  # (num_groups, emsize)
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
