# import torch
#
#
# def pack(real, imag):
#     return torch.stack([real, imag], dim=-1)
#
#
# def unpack(x):
#     return x[..., 0], x[..., 1]
#
#
# def conj(x):
#     return torch.stack([x[..., 0], -x[..., 1]], dim=-1)
#
#
# def ones(shape, dtype=torch.float32, device=torch.device('cpu')):
#     return torch.stack([torch.ones(shape, dtype=dtype, device=device),
#                         torch.zeros(shape, dtype=dtype, device=device)], dim=-1)
#
#
# def eye(K):
#     return torch.stack([torch.eye(K), torch.zeros((K, K))], dim=-1)
#
#
# def abs2(x):
#     return x[..., -1] ** 2 + x[..., -2] ** 2
#
#
# def multiply(x, y):
#     x_real, x_imag = unpack(x)
#     y_real, y_imag = unpack(y)
#     return torch.stack([x_real * y_real - x_imag * y_imag, x_imag * y_real + x_real * y_imag], dim=-1)
#
#
# def mul_with_func(x, y, func):
#     x_real, x_imag = unpack(x)
#     y_real, y_imag = unpack(y)
#     xr_yr = func(x_real, y_real)
#     xr_yi = func(x_real, y_imag)
#     xi_yr = func(x_imag, y_real)
#     xi_yi = func(x_imag, y_imag)
#     real = xr_yr - xi_yi
#     imag = xr_yi + xi_yr
#     return torch.stack([real, imag], dim=-1)
# util/complex.py (现代化API版本)

import torch


def conj(x: torch.Tensor) -> torch.Tensor:
    """计算复数张量的共轭"""
    return torch.conj(x)


def abs2(x: torch.Tensor) -> torch.Tensor:
    """计算复数张量的模长的平方"""
    return x.real ** 2 + x.imag ** 2


def multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算两个复数张量的乘积"""
    return x * y

# 注意：旧的 mul_with_func, pack, unpack, ones, eye 等函数
# 在新的原生复数张量体系下已不再需要或可以通过更简单的方式实现，
# 我们将它们移除以简化代码。
