from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class Regularizer(ABC):
    """Abstract base class for regularization terms."""

    @abstractmethod
    def __call__(self, kappa: npt.NDArray[np.float64]) -> float:
        """Calculate regularization value.

        Args:
            kappa: Parameter array to regularize

        Returns:
            Regularization value
        """
        pass

    @abstractmethod
    def gradient(self, kappa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of regularization term.

        Args:
            kappa: Parameter array to regularize

        Returns:
            Gradient of regularization with respect to kappa
        """
        pass


class ValueRegularizer(Regularizer):
    """Regularizer that penalizes the Lp norm of the parameter values.

    The functional is defined as: R(κ) = (λ/p) ∫|κ|ᵖ dx

    Args:
        lambda_reg: Regularization strength
        p: The power in the Lp norm (default: 2)
    """

    def __init__(self, lambda_reg: float, p: float = 2):
        self.lambda_reg = lambda_reg
        self.p = p

    def __call__(self, kappa: npt.NDArray[np.float64]) -> float:
        """Calculate the Lp norm regularization value."""
        return (self.lambda_reg / self.p) * np.sum(np.abs(kappa) ** self.p)

    def gradient(self, kappa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of the Lp norm regularization.

        For Lp norm regularization, the gradient is: λ |κ|^(p-1) sign(κ)
        """
        # Handle special case of p=1 to avoid numerical issues
        if self.p == 1:
            return self.lambda_reg * np.sign(kappa)

        return self.lambda_reg * np.sign(kappa) * np.abs(kappa) ** (self.p - 1)


class GradientRegularizer(Regularizer):
    """Regularizer that penalizes the Lp norm of the parameter gradients.

    The functional is defined as: R(κ) = (λ/p) ∫|∇κ|ᵖ dx

    This enforces smoothness in the parameter κ.

    Args:
        lambda_reg: Regularization strength
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        p: The power in the Lp norm (default: 2)
        epsilon: Small value to prevent division by zero for p < 2 (default: 1e-10)
    """

    def __init__(
        self,
        lambda_reg: float,
        dx: float,
        dy: float,
        p: float = 2,
        epsilon: float = 1e-10,
    ):
        self.lambda_reg = lambda_reg
        self.dx = dx
        self.dy = dy
        self.p = p
        self.epsilon = epsilon  # To prevent division by zero for p < 2

    def __call__(self, kappa: npt.NDArray[np.float64]) -> float:
        """Calculate the gradient Lp norm regularization value."""
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        dx = np.gradient(kappa_2d, axis=0) / self.dx
        dy = np.gradient(kappa_2d, axis=1) / self.dy

        # Calculate |∇κ|^p
        grad_magnitude_p = (dx**2 + dy**2) ** (self.p / 2)

        return (self.lambda_reg / self.p) * np.sum(grad_magnitude_p)

    def gradient(self, kappa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of the gradient Lp norm regularization.

        For p=2, this is -λ∆κ (negative Laplacian).
        For general p, this is -λ div(|∇κ|^(p-2) ∇κ).
        """
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        dx = np.gradient(kappa_2d, axis=0) / self.dx
        dy = np.gradient(kappa_2d, axis=1) / self.dy

        # Special case: p=2 (standard Laplacian case)
        if self.p == 2:
            reg_grad = -self.lambda_reg * (
                np.gradient(dx, axis=0) / self.dx + np.gradient(dy, axis=1) / self.dy
            )
            return reg_grad.flatten()

        # General p case: -λ div(|∇κ|^(p-2) ∇κ)
        # Calculate |∇κ|^(p-2)
        grad_magnitude = np.sqrt(
            dx**2 + dy**2 + self.epsilon
        )  # Add epsilon to prevent division by zero
        weight = grad_magnitude ** (self.p - 2)

        # Calculate div(|∇κ|^(p-2) ∇κ)
        weighted_dx = weight * dx
        weighted_dy = weight * dy

        div_weighted_grad = (
            np.gradient(weighted_dx, axis=0) / self.dx
            + np.gradient(weighted_dy, axis=1) / self.dy
        )

        return (-self.lambda_reg * div_weighted_grad).flatten()


class TotalVariationRegularizer(Regularizer):
    """Total Variation regularizer.

    A special case of the gradient regularizer with p=1: R(κ) = λ ∫|∇κ| dx
    Uses a smoothed approximation for numerical stability.

    Args:
        lambda_reg: Regularization strength
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        epsilon: Smoothing parameter (default: 1e-6)
    """

    def __init__(self, lambda_reg: float, dx: float, dy: float, epsilon: float = 1e-6):
        self.lambda_reg = lambda_reg
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon  # Smoothing parameter

    def __call__(self, kappa: npt.NDArray[np.float64]) -> float:
        """Calculate the Total Variation regularization value."""
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        dx = np.gradient(kappa_2d, axis=0) / self.dx
        dy = np.gradient(kappa_2d, axis=1) / self.dy

        # Smoothed TV: ∫√(|∇κ|² + ε) dx
        return self.lambda_reg * np.sum(np.sqrt(dx**2 + dy**2 + self.epsilon))

    def gradient(self, kappa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate gradient of the Total Variation regularization."""
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        dx = np.gradient(kappa_2d, axis=0) / self.dx
        dy = np.gradient(kappa_2d, axis=1) / self.dy

        # Calculate 1/√(|∇κ|² + ε)
        grad_magnitude = np.sqrt(dx**2 + dy**2 + self.epsilon)
        weight = 1.0 / grad_magnitude

        # Calculate div(∇κ/|∇κ|)
        weighted_dx = weight * dx
        weighted_dy = weight * dy

        div_weighted_grad = (
            np.gradient(weighted_dx, axis=0) / self.dx
            + np.gradient(weighted_dy, axis=1) / self.dy
        )

        return (-self.lambda_reg * div_weighted_grad).flatten()


class DenoiserRegularizer(Regularizer):
    """Neural network denoiser-based regularizer (RED): λ/2 x^T(x - f(x)) for f(x) = denoiser(x)"""

    def __init__(self, denoiser: nn.Module, lambda_reg: float, device: str = "cpu"):
        self.denoiser = denoiser
        self.lambda_reg = lambda_reg
        self.device = device
        self.denoiser.eval()  # Ensure denoiser is in evaluation mode

    def __call__(self, kappa: npt.NDArray[np.float64]) -> float:
        with torch.no_grad():
            kappa_torch = torch.from_numpy(kappa).float().to(self.device)
            if kappa_torch.ndim == 1:
                side = int(np.sqrt(len(kappa)))
                kappa_2d = kappa_torch.reshape(1, 1, side, side)
            elif kappa_torch.ndim == 2:
                # add batch and channel dimensions
                kappa_2d = kappa_torch.unsqueeze(0).unsqueeze(0)

            # could have channel dimension, but missing batch dimension
            elif kappa_torch.ndim == 3:
                kappa_2d = kappa_torch.unsqueeze(0)
            else:
                kappa_2d = kappa_torch

            denoised = self.denoiser(kappa_2d).squeeze(0).squeeze(0).cpu().numpy()
            # Energy: λ/2 x^T (x - f(x))
            energy = 0.5 * self.lambda_reg * np.sum(kappa * (kappa - denoised))
        return energy

    def gradient(self, kappa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Revised gradient computation using autograd to correctly differentiate the energy
        kappa_torch = torch.from_numpy(kappa).float().to(self.device)
        kappa_torch.requires_grad_(True)
        if kappa_torch.ndim == 1:
            side = int(np.sqrt(len(kappa)))
            kappa_2d = kappa_torch.reshape(1, 1, side, side)
        elif kappa_torch.ndim == 2:
            kappa_2d = kappa_torch.unsqueeze(0).unsqueeze(0)
        elif kappa_torch.ndim == 3:
            kappa_2d = kappa_torch.unsqueeze(0)
        else:
            kappa_2d = kappa_torch
        # Compute denoiser output with gradient tracking
        denoised = self.denoiser(kappa_2d).squeeze(0).squeeze(0)
        # analytical RED gradient (x - f(x))
        grad = kappa_torch - denoised
        return grad.cpu().numpy()
