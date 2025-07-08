import typing as tp
from dataclasses import dataclass

from pfns import base_config
from pfns.model import encoders, transformer
from pfns.model.bar_distribution import BarDistribution
from pfns.model.criterions import BarDistributionConfig, CrossEntropyConfig
from pfns.model.encoders import StyleEncoderConfig

from torch import nn


@dataclass(frozen=True)
class TransformerConfig(base_config.BaseConfig):
    criterion: CrossEntropyConfig | BarDistributionConfig
    encoder: tp.Optional[encoders.EncoderConfig] = (
        None  # todo add back in as config, currently only supporting standard encoder
    )
    y_encoder: tp.Optional[encoders.EncoderConfig] = (
        None  # todo add back in as config, currently only supporting standard encoder
    )
    style_encoder: tp.Optional[StyleEncoderConfig] = None
    y_style_encoder: tp.Optional[StyleEncoderConfig] = None
    decoder_dict: tp.Dict[str, base_config.BaseTypes] | None = None
    emsize: int = 200
    nhid: int = 200
    nlayers: int = 6
    nhead: int = 2
    features_per_group: int = 1
    attention_between_features: bool = True
    model_extra_args: tp.Dict[str, base_config.BaseTypes] | None = None

    def create_model(self) -> transformer.TableTransformer:
        # Resolve criterion
        criterion = self.criterion.get_criterion()

        # Determine n_out based on the resolved criterion
        if isinstance(criterion, BarDistribution):
            n_out = criterion.num_bars
        elif isinstance(criterion, nn.CrossEntropyLoss):
            n_out = criterion.weight.shape[0]
        else:
            raise ValueError(f"Criterion {criterion} not supported")

        decoder_dict = (
            self.decoder_dict if self.decoder_dict else {"standard": (None, n_out)}
        )

        if self.encoder is not None:
            encoder = self.encoder.create_encoder(
                features=self.features_per_group, emsize=self.emsize
            )
        else:
            encoder = None

        if self.y_encoder is not None:
            y_encoder = self.y_encoder.create_encoder(features=1, emsize=self.emsize)
        else:
            y_encoder = None

        if self.style_encoder is not None:
            style_encoder = self.style_encoder.create_encoder(self.emsize)
        else:
            style_encoder = None

        if self.y_style_encoder is not None:
            y_style_encoder = self.y_style_encoder.create_encoder(self.emsize)
        else:
            y_style_encoder = None

        model = transformer.TableTransformer(
            encoder=encoder,
            y_encoder=y_encoder,
            features_per_group=self.features_per_group,
            decoder_dict=decoder_dict,
            ninp=self.emsize,
            nhid=self.nhid,
            nlayers=self.nlayers,
            nhead=self.nhead,
            attention_between_features=self.attention_between_features,
            style_encoder=style_encoder,
            y_style_encoder=y_style_encoder,
            batch_first=True,  # model is batch_first by default now
            **(self.model_extra_args or {}),
        )
        model.criterion = criterion
        return model
