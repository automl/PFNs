import torch
from pfns.model import bar_distribution

from pytest import approx


def test_cdf_out_of_bounds():
    logits = torch.tensor([0.05, 0.05, 0.1, 0.3, 0.5]).log()
    d = bar_distribution.BarDistribution(
        borders=torch.tensor([0.0, 1.0, 2.0, 2.5, 4.0, 6.0])
    )

    # outside range
    assert d.cdf(logits, torch.tensor([-1.0]))[0].item() == approx(0.0)
    assert d.cdf(logits, torch.tensor([7.0]))[0].item() == approx(1.0)

    # on borders
    assert d.cdf(logits, torch.tensor([0.0]))[0].item() == approx(0.0)
    assert d.cdf(logits, torch.tensor([1.0]))[0].item() == approx(0.05)
    assert d.cdf(logits, torch.tensor([2.5]))[0].item() == approx(0.2)
    assert d.cdf(logits, torch.tensor([6.0]))[0].item() == approx(1.0)

    # inside bucket
    assert d.cdf(logits, torch.tensor([1.5]))[0].item() == approx(0.075)


def test_move_to_larger():
    logits = torch.arange(99).float()
    old_d = bar_distribution.BarDistribution(borders=torch.linspace(0, 1, 100))
    new_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 1000))
    new_logits = old_d.get_probs_for_different_borders(logits, new_d.borders).log()
    assert old_d.median(logits).item() == approx(new_d.median(new_logits).item())
    assert old_d.mean(logits).item() == approx(new_d.mean(new_logits).item(), abs=1e-4)


def test_average_bar_distributions_into_different_one():
    num_bars = [100, 80, 10, 5]
    logits = [torch.arange(nb - 1).float() for nb in num_bars]
    bar_dists = [
        bar_distribution.BarDistribution(borders=torch.linspace(-1, -0.5, num_bars[0])),
        bar_distribution.BarDistribution(borders=torch.linspace(0, 2, num_bars[1])),
        bar_distribution.BarDistribution(borders=torch.linspace(1, 3, num_bars[2])),
        bar_distribution.BarDistribution(borders=torch.linspace(2, 3, num_bars[3])),
    ]

    new_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 100))
    new_logits = new_d.average_bar_distributions_into_this(bar_dists, logits)

    assert new_d.cdf(new_logits, torch.tensor([-1.0])).item() == approx(0.0)
    assert new_d.cdf(new_logits, torch.tensor([0.0])).item() == approx(0.25)
    assert new_d.cdf(new_logits, torch.tensor([3.0])).item() == approx(1.0)

    new_small_d = bar_distribution.BarDistribution(borders=torch.linspace(-1, 2, 10))
    new_small_logits = new_small_d.average_bar_distributions_into_this(
        bar_dists, logits
    )

    assert new_small_d.cdf(new_small_logits, torch.tensor([-1.0])).item() == approx(0.0)
    assert new_small_d.cdf(new_small_logits, torch.tensor([0.0])).item() == approx(0.25)
    assert new_small_d.cdf(new_small_logits, torch.tensor([2.0])).item() == approx(1.0)
    pos = torch.tensor([new_small_d.borders[-2]])
    assert new_small_d.cdf(new_small_logits, pos).item() == approx(
        sum([bd.cdf(l, pos)[0].item() for bd, l in zip(bar_dists, logits)]) / 4
    )


def test_entropy():
    prob = torch.tensor([0.1, 0.2, 0.3, 0.4])
    d = bar_distribution.BarDistribution(borders=torch.linspace(0, 1, 5))
    logits = prob.log()
    # Manual: H = -sum p * log(p/width) = -sum p log p + sum p log width
    widths = d.borders[1:] - d.borders[:-1]
    manual = -(prob * prob.log()).sum() + (prob * widths.log()).sum()
    val = d.entropy(logits)
    assert val.item() == approx(manual.item(), abs=1e-6)


def test_entropy_full_support():
    # Use three buckets so we have two half-normals and one interior uniform
    borders = torch.tensor([-1.0, 0.0, 1.0, 3.0])
    d = bar_distribution.FullSupportBarDistribution(borders=borders)
    prob = torch.tensor([0.2, 0.5, 0.3])
    logits = prob.log()
    # Component entropies: left/right half-normal at scales set by bucket widths
    left_hn = d.halfnormal_with_p_weight_before(d.bucket_widths[0])
    right_hn = d.halfnormal_with_p_weight_before(d.bucket_widths[-1])
    H_components = torch.tensor(
        [
            left_hn.entropy(),
            torch.log(
                d.bucket_widths[1]
            ),  # no negation needed as bucket_widths = 1/prob
            right_hn.entropy(),
        ]
    )
    manual = (prob * (-prob.log() + H_components)).sum()
    val = d.entropy(logits)
    assert val.item() == approx(manual.item(), abs=1e-6)
