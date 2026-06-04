# """
# Refer to
# https://github.com/AndreiDavydov/Poisson_Denoiser/blob/master/pydl/nnLayers/functional/functional.py
# under MIT Licence (copyright: Andrei Davydov)
# """
# import torch
#
# from util import complex
#
#
# def autocorrelation1d_symmetric(h):
#     """Compute autocorrelation of a symmetric signal along the last dimension"""
#     Fhsq = complex.abs2(torch.rfft(h, 1))
#     a = torch.irfft(torch.stack([Fhsq, torch.zeros_like(Fhsq)], dim=-1), 1, signal_sizes=(h.shape[-1],))
#     return a / a.max()
#
#
# def compute_weighting_for_tapering(h):
#     """Compute autocorrelation of a symmetric signal along the last two dimension"""
#     h_proj0 = h.sum(dim=-2, keepdims=False)
#     autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2)
#     h_proj1 = h.sum(dim=-1, keepdims=False)
#     autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1)
#     return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)
#
#
# def edgetaper3d(img, psf):
#     """
#     Edge-taper an image with a depth-dependent PSF
#
#     Args:
#         img: (B x C x H x W)
#         psf: 3d rotationally-symmetric psf (B x C x D x H x W) (i.e. continuous at boundaries)
#
#     Returns:
#         Edge-tapered 3D image
#     """
#     assert (img.dim() == 4)
#     assert (psf.dim() == 5)
#     psf = psf.mean(dim=-3)
#     alpha = compute_weighting_for_tapering(psf)
#     blurred_img = torch.irfft(
#         complex.multiply(torch.rfft(img, 2), torch.rfft(psf, 2)), 2, signal_sizes=img.shape[-2:]
#     )
#     return alpha * img + (1 - alpha) * blurred_img

# util/edgetaper.py (现代化API最终版)

import torch
from . import complex  # 使用相对导入，更稳健


def autocorrelation1d_symmetric(h):
    """使用现代torch.fft API计算对称信号的自相关"""
    # 使用 torch.fft.rfft 进行1D傅里叶变换
    Fh = torch.fft.rfft(h, dim=-1)
    Fhsq = complex.abs2(Fh)

    # 使用 torch.fft.irfft 进行逆变换
    a = torch.fft.irfft(Fhsq, n=h.shape[-1], dim=-1)

    # a.max() 可能为0，增加一个小的eps防止除零错误
    return a / (a.max() + 1e-8)


def compute_weighting_for_tapering(h):
    """计算用于边缘锥化的权重"""
    h_proj0 = h.sum(dim=-2, keepdim=False)
    autocorr_h_proj0 = autocorrelation1d_symmetric(h_proj0).unsqueeze(-2)
    h_proj1 = h.sum(dim=-1, keepdim=False)
    autocorr_h_proj1 = autocorrelation1d_symmetric(h_proj1).unsqueeze(-1)
    return (1 - autocorr_h_proj0) * (1 - autocorr_h_proj1)


def edgetaper3d(img, psf):
    """
    对图像进行边缘锥化处理。
    """
    assert (img.dim() == 4)
    assert (psf.dim() == 5)

    # .mean(dim=-3) 在多通道时可能有问题，我们对所有深度层取平均
    psf_mean_depth = psf.mean(dim=2)  # (B, C, H, W)

    alpha = compute_weighting_for_tapering(psf_mean_depth)

    # 使用现代torch.fft API进行2D卷积
    Fimg = torch.fft.rfft2(img, dim=(-2, -1))
    Fpsf = torch.fft.rfft2(psf_mean_depth, dim=(-2, -1))

    blurred_img = torch.fft.irfft2(
        complex.multiply(Fimg, Fpsf), s=img.shape[-2:], dim=(-2, -1)
    )

    return alpha * img + (1 - alpha) * blurred_img
