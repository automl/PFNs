import pytest
import torch
import torch.nn as nn

from pfns.model import encoders, transformer
from pfns.model.transformer import (
    DEFAULT_EMSIZE,
    isolate_torch_rng,
    PerFeatureTransformer,
)


class SimpleStyleEncoder(nn.Module):
    def __init__(self, ninp):
        super().__init__()
        self.linear = nn.Linear(5, ninp)

    def forward(self, x):
        if x.ndim == 2:
            return self.linear(x)
        elif x.ndim == 3:  # shape: batch size x num_features x style_dim
            return self.linear(x).sum(dim=1)


@pytest.fixture
def sample_data():
    batch_size = 4
    seq_len_train = 10
    seq_len_test = 5
    num_features = 3

    train_x = torch.randn(seq_len_train, batch_size, num_features)
    train_y = torch.randn(seq_len_train, batch_size, 1)
    test_x = torch.randn(seq_len_test, batch_size, num_features)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "batch_size": batch_size,
        "seq_len_train": seq_len_train,
        "seq_len_test": seq_len_test,
        "num_features": num_features,
    }


def test_transformer_init():
    """Test basic initialization of the transformer."""
    transformer = PerFeatureTransformer(ninp=64, nhead=4, nhid=256, nlayers=6)

    assert transformer.ninp == 64
    assert transformer.nhead == 4
    assert transformer.nhid == 256
    assert len(transformer.transformer_layers.layers) == 6


def test_transformer_forward(sample_data):
    """Test basic forward pass with default parameters."""
    transformer = PerFeatureTransformer(ninp=32, nhead=2, nhid=64, nlayers=2)

    output = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )


def test_transformer_seed_behavior(sample_data):
    """Test that using the same seed produces the same outputs."""
    seed = 42

    # Create two transformers with the same seed
    transformer = PerFeatureTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        feature_positional_embedding="normal_rand_vec",
        seed=seed,
    )

    # Run both models
    output1 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    output2 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    # Outputs should be identical with the same seed
    assert torch.allclose(output1, output2)

    # Create a transformer with a different seed
    transformer3 = PerFeatureTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        feature_positional_embedding="normal_rand_vec",
        seed=seed + 1,
    )

    output3 = transformer3(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    # Outputs should be different with a different seed
    assert not torch.allclose(output1, output3)


def test_isolate_torch_rng():
    """Test that isolate_torch_rng properly isolates the random number generator state."""

    # Generate a random tensor before using isolate_torch_rng
    torch.manual_seed(123)
    tensor_before1 = torch.rand(5)
    tensor_before2 = torch.rand(5)
    torch.manual_seed(123)
    tensor_before_1 = torch.rand(5)

    # Sanity check that seeding works
    assert torch.allclose(tensor_before1, tensor_before_1)

    # Use isolate_torch_rng with a different seed
    with isolate_torch_rng(seed=123, device=torch.device("cpu")):
        # Check if seed correctly set
        isolated_tensor1 = torch.rand(5)
        isolated_tensor2 = torch.rand(5)

        # These should be identical
        assert torch.allclose(isolated_tensor1, tensor_before1)
        assert torch.allclose(isolated_tensor1, tensor_before_1)
        assert torch.allclose(isolated_tensor2, tensor_before2)

    # RNG state restorement
    tensor_after_2 = torch.rand(5)

    assert torch.allclose(tensor_before2, tensor_after_2)
    assert torch.allclose(tensor_after_2, isolated_tensor2)


def test_feature_positional_embeddings(sample_data):
    """Test different feature positional embedding options."""
    embedding_types = [
        "normal_rand_vec",
        "uni_rand_vec",
        "learned",
        "subspace",
        None,
    ]

    for emb_type in embedding_types:
        transformer = PerFeatureTransformer(
            ninp=32,
            nhead=2,
            nhid=64,
            nlayers=2,
            feature_positional_embedding=emb_type,
            seed=42,
        )

        output = transformer(
            x=sample_data["train_x"],
            y=sample_data["train_y"],
            test_x=sample_data["test_x"],
        )

        assert isinstance(output, torch.Tensor)
        assert output.shape == (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )


def test_features_per_group(sample_data):
    """Test the features_per_group parameter."""
    # Set features_per_group=3 to match the number of features in sample data
    transformer = PerFeatureTransformer(
        ninp=32, nhead=2, nhid=64, nlayers=2, features_per_group=3
    )

    output = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )


@torch.inference_mode()
def test_cache_trainset_representation(sample_data):
    """Test caching of trainset representations."""
    transformer = PerFeatureTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        cache_trainset_representation=True,
    )

    # First forward pass should cache the representations
    output1 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    # Second forward pass should use the cached representations
    output2 = transformer(x=None, y=None, test_x=sample_data["test_x"])

    assert torch.allclose(output1, output2)

    # Clear cache and results should be different
    transformer.empty_trainset_representation_cache()

    # After clearing, we need to provide train data again
    output3 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert torch.allclose(output1, output3)  # Should be deterministic


def test_decoder_dict(sample_data):
    """Test custom decoder dictionary."""

    class CustomDecoder(nn.Module):
        def __init__(self, ninp, nhid, nout):
            super().__init__()
            self.linear1 = nn.Linear(ninp, nhid)
            self.linear2 = nn.Linear(nhid, nout)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    decoder_dict = {
        "standard": (None, 1),  # Default MLP
        "custom": (CustomDecoder, 2),  # Custom decoder with 2 outputs
    }

    transformer = PerFeatureTransformer(
        ninp=32, nhead=2, nhid=64, nlayers=2, decoder_dict=decoder_dict
    )

    # Get all outputs
    outputs = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        only_return_standard_out=False,
    )

    assert isinstance(outputs, dict)
    assert "standard" in outputs
    assert "custom" in outputs
    assert outputs["standard"].shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )
    assert outputs["custom"].shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        2,
    )


def test_style_encoder(sample_data):
    """Test the style encoder functionality."""
    style_encoder = SimpleStyleEncoder(32)

    transformer = PerFeatureTransformer(
        ninp=32, nhead=2, nhid=64, nlayers=2, style_encoder=style_encoder
    )

    # Create style vectors for each batch
    style = torch.randn(sample_data["batch_size"], 5)

    output = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=style,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )

    # Test per-feature style vectors
    feature_style = torch.randn(
        sample_data["batch_size"], sample_data["num_features"], 5
    )

    output2 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=feature_style,
    )

    assert isinstance(output2, torch.Tensor)
    assert output2.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )


def test_y_style_encoder(sample_data):
    """Test the y_style encoder functionality."""
    style_encoder = SimpleStyleEncoder(32)

    transformer = PerFeatureTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        style_encoder=style_encoder,
        y_style_encoder=style_encoder,
    )

    # Create y_style vectors for each batch
    y_style = torch.randn(sample_data["batch_size"], 5)

    output = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        y_style=y_style,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )

    # Test with both style and y_style
    style = torch.randn(sample_data["batch_size"], 5)

    output2 = transformer(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=style,
        y_style=y_style,
    )

    assert isinstance(output2, torch.Tensor)
    assert output2.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )

    # Test per-feature style with both style and y_style
    feature_style = torch.randn(
        sample_data["batch_size"], sample_data["num_features"], 5
    )

    transformer_per_feature = PerFeatureTransformer(
        ninp=32,
        nhead=2,
        nhid=64,
        nlayers=2,
        style_encoder=style_encoder,
        y_style_encoder=style_encoder,
    )

    output3 = transformer_per_feature(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=feature_style,
        y_style=y_style,
    )

    assert isinstance(output3, torch.Tensor)
    assert output3.shape == (
        sample_data["seq_len_test"],
        sample_data["batch_size"],
        1,
    )


@pytest.mark.parametrize(
    "multiquery_item_attention_for_test_set",
    [False, True],
)
@torch.inference_mode()
def test_separate_train_inference(multiquery_item_attention_for_test_set):
    model = transformer.PerFeatureTransformer(
        encoder=encoders.SequentialEncoder(
            encoders.InputNormalizationEncoderStep(
                normalize_on_train_only=True,
                normalize_to_ranking=False,
                normalize_x=True,
                remove_outliers=True,
            ),  # makes it more interesting
            encoders.LinearInputEncoderStep(
                num_features=1,
                emsize=transformer.DEFAULT_EMSIZE,
                in_keys=["main"],
                out_keys=["output"],
            ),
        ),
    )

    for p in model.parameters():
        p += 0.01  # make it more interesting, not anymore mean 0

    model.feature_positional_embedding = None  # 'subspace'
    for layer in model.transformer_layers.layers:
        layer.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )

    model.cache_trainset_representation = True
    model.reset_save_peak_mem_factor(None)
    model.empty_trainset_representation_cache()

    device = "cpu"

    n_train = 10
    n_features = 10
    n_test = 3
    batch_size = 2
    x_train = torch.normal(
        0.0,
        2.0,
        size=(n_train, batch_size, n_features),
        device=device,
    )
    y = (x_train[:, :, :1] > 1.0).float().to(device).to(torch.float)
    x_test = torch.normal(
        0.0,
        1.0,
        size=(n_test, batch_size, n_features),
        device=device,
    )

    torch.manual_seed(12345)
    model(x_train, y[:n_train])
    logits1 = model(x_test, None)

    torch.manual_seed(12345)
    logits1a = model(x_train, y, x_test)

    assert logits1.float() == pytest.approx(
        logits1a.float(), abs=1e-5
    ), f"{logits1} != {logits1a}"
