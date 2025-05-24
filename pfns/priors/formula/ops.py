import math

import torch


# all ops are implemented in torch mapping vectors of length n to vectors of length n
### binary operations


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def gate(x, y):
    return (x > 0) * y


def power(x, y):
    return x.abs() ** y


def rr_ad_gate(x, y):
    bound_index = torch.randint(len(x), (1,)).squeeze()
    return (x > x[bound_index]) * y


binary_ops = {
    "add": add,
    "mul": mul,
    "gate": gate,
    "power": power,
    "rr_ad_gate": rr_ad_gate,
}

### unary operations


def ident(x):
    return x


def absolute(x):
    return x.abs()


def inv(x):
    return 1 / x


def sin(x):
    return torch.sin(x * 4 * math.pi)


def exp(x):
    return torch.exp(x)


def log(x):
    return torch.log(x.abs())


def sqrt(x):
    return torch.sqrt(x.abs())


def square(x):
    return x**2


def sigmoid(x):
    return torch.sigmoid(x)


def cubic(x):
    return x**3


# def relative_noise__random(x):
#     return x * torch.randn_like(x)

unary_ops = {
    "ident": ident,
    "abs": absolute,
    "inv": inv,
    "sin": sin,
    "exp": exp,
    "log": log,
    "sqrt": sqrt,
    "square": square,
    "sigmoid": sigmoid,
    "cubic": cubic,
    # 'rr_rel_noise': relative_noise__random,
}
