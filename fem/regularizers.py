from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from typing import List, Optional


class Regularizer(ABC):
    """Abstract base class for regularization terms."""

    @abstractmethod
    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> float:
        """Calculate regularization value.

        Args:
            kappa: Parameter array to regularize
            lambda_reg: Optional override for regularization strength

        Returns:
            Regularization value
        """
        pass

    @abstractmethod
    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of regularization term.

        Args:
            kappa: Parameter array to regularize
            lambda_reg: Optional override for regularization strength

        Returns:
            Gradient of regularization with respect to kappa
        """
        pass

    def update_lambda(self, lambda_reg: float) -> None:
        """Update the regularization strength.

        Args:
            lambda_reg: New regularization strength
        """
        # Base implementation might not do anything if lambda is handled differently
        # or within __call__/gradient, but subclasses can override.
        # Specific regularizers like ValueRegularizer already store lambda_reg.
        pass


class ValueRegularizer(Regularizer):
    """Regularizer that penalizes the Lp norm of the parameter values.

    The functional is defined as:
        R(κ) = (λ/p) ∫ |κ|ᵖ dx
    and should be discretized as:
        R(κ) ≈ (λ/p) * ΔA * Σ_i |κ_i|ᵖ,
    with the gradient:
        ∇R(κ)_i = λ * ΔA * |κ_i|^(p-1) * sign(κ_i).
    """

    def __init__(
        self, lambda_reg: float, p: float = 2, dx: float = 1.0, dy: float = 1.0
    ):
        self.lambda_reg = lambda_reg
        self.p = p
        self.dx = dx
        self.dy = dy

    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: float = None
    ) -> float:
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg
        cell_area = self.dx * self.dy
        return (self.lambda_reg / self.p) * np.sum(np.abs(kappa) ** self.p) * cell_area

    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: float = None
    ) -> npt.NDArray[np.float64]:
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg
        cell_area = self.dx * self.dy
        if self.p == 1:
            return self.lambda_reg * np.sign(kappa) * cell_area
        return (
            self.lambda_reg * np.sign(kappa) * np.abs(kappa) ** (self.p - 1) * cell_area
        )

    def update_lambda(self, lambda_reg: float) -> None:
        """Update the regularization strength."""
        self.lambda_reg = lambda_reg


class GradientRegularizer(Regularizer):
    """Regularizer that penalizes the Lp norm of the parameter gradients.

    The functional is defined as R(κ) = (λ/p) ∫|∇κ|ᵖ dx

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

    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: float = None
    ) -> float:
        """Calculate the gradient Lp norm regularization value."""
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        dx = np.gradient(kappa_2d, axis=0) / self.dx
        dy = np.gradient(kappa_2d, axis=1) / self.dy

        # Calculate |∇κ|^p
        grad_magnitude_p = (dx**2 + dy**2) ** (self.p / 2)

        return (self.lambda_reg / self.p) * np.sum(grad_magnitude_p)

    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: float = None
    ) -> npt.NDArray[np.float64]:
        """Calculate gradient of the gradient Lp norm regularization.

        For p=2, this is -λ∆κ (negative Laplacian).
        For general p, this is -λ div(|∇κ|^(p-2) ∇κ).
        """
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg
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

    def update_lambda(self, lambda_reg: float) -> None:
        """Update the regularization strength."""
        self.lambda_reg = lambda_reg


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

    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> float:
        """Calculate the Total Variation regularization value."""
        current_lambda = lambda_reg if lambda_reg is not None else self.lambda_reg
        kappa_2d = kappa.reshape((int(np.sqrt(len(kappa))), -1))
        # Use finite differences for gradient approximation
        # Ensure we handle boundary conditions appropriately if needed,
        # np.gradient uses central differences in the interior and first differences at boundaries
        grad_x = np.gradient(kappa_2d, axis=0) / self.dx
        grad_y = np.gradient(kappa_2d, axis=1) / self.dy

        # Smoothed TV: ∫√(|∇κ|² + ε) dx dy
        # Integrate over the area by multiplying by cell area dx * dy
        cell_area = self.dx * self.dy
        return current_lambda * np.sum(
            np.sqrt(grad_x**2 + grad_y**2 + self.epsilon) * cell_area
        )

    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> npt.NDArray[np.float64]:
        current_lambda = lambda_reg if lambda_reg is not None else self.lambda_reg
        # Convert numpy array to torch tensor and enable gradient tracking.
        kappa_tensor = torch.tensor(
            kappa.reshape((int(np.sqrt(len(kappa))), -1)),
            dtype=torch.float32,
            requires_grad=True,
        )

        # Compute the total variation regularizer using torch operations:
        # the finite-difference TV using torch.gradient similar to np.gradient
        # include multiplication by dx*dy for the integration measure.
        grad_x = torch.gradient(kappa_tensor, spacing=self.dx)[0] / self.dx
        grad_y = torch.gradient(kappa_tensor, spacing=self.dy)[1] / self.dy
        tv = torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon))
        tv = current_lambda * tv * self.dx * self.dy

        # Trigger the backward pass:
        tv.backward()

        # Extract the gradient and convert to numpy array:
        grad_numpy = kappa_tensor.grad.cpu().detach().numpy().flatten()
        return grad_numpy


class DenoiserRegularizer(Regularizer):
    """Neural network denoiser-based regularizer (RED): λ/2 x^T(x - f(x)) for f(x) = denoiser(x)"""

    def __init__(
        self,
        denoiser: nn.Module,
        lambda_reg: float,
        device: str = "cpu",
        norm_min: Optional[float] = None,
        norm_max: Optional[float] = None,
    ):
        self.denoiser = denoiser
        self.lambda_reg = lambda_reg
        self.device = device
        self.denoiser.eval()  # Ensure denoiser is in evaluation mode
        self.norm_min = norm_min
        self.norm_max = norm_max

    def _preprocess(self, kappa: npt.NDArray[np.float64]) -> torch.Tensor:
        """Convert numpy array to torch tensor, normalize, and set requires_grad."""
        kappa_torch = torch.from_numpy(kappa).float().to(self.device)
        if self.norm_min is not None and self.norm_max is not None:
            kappa_torch = (kappa_torch - self.norm_min) / (
                self.norm_max - self.norm_min
            )

        if kappa_torch.ndim == 1:
            side = int(np.sqrt(len(kappa)))
            kappa_2d = kappa_torch.reshape(1, 1, side, side)
        elif kappa_torch.ndim == 2:
            kappa_2d = kappa_torch.unsqueeze(0).unsqueeze(0) # Add batch and channel
        elif kappa_torch.ndim == 3: # Assume (C, H, W)
            kappa_2d = kappa_torch.unsqueeze(0) # Add batch
        else: # Assume (B, C, H, W)
            kappa_2d = kappa_torch

        kappa_2d.requires_grad_(True)
        return kappa_2d

    def _postprocess(self, denoised_norm: torch.Tensor) -> npt.NDArray[np.float64]:
        """Denormalize and convert back to numpy."""
        denoised = denoised_norm.squeeze().detach().cpu().numpy()
        if self.norm_min is not None and self.norm_max is not None:
            denoised = denoised * (self.norm_max - self.norm_min) + self.norm_min
        return denoised.flatten()


    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> float:
        """Calculate the RED regularization value: λ/2 ||κ - f(κ)||_2^2"""
        current_lambda = lambda_reg if lambda_reg is not None else self.lambda_reg
        with torch.no_grad():
            kappa_2d_norm = self._preprocess(kappa)
            # Detach input before passing to denoiser if not computing gradient via autograd
            denoised_norm = self.denoiser(kappa_2d_norm.detach())
            denoised = self._postprocess(denoised_norm)

            # R(x) = λ/2 ||x - f(x)||^2
            diff = kappa - denoised
            energy = 0.5 * current_lambda * np.dot(diff, diff)
            # Note: Removed the original x^T(x-f(x)) formulation as ||x-f(x)||^2 is more common for RED.
            # If the original formulation is intended, this needs adjustment.
        return energy

    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> npt.NDArray[np.float64]:
        """Calculate the RED gradient: λ * (κ - f(κ)) with respect to the physical kappa"""
        # This uses the common approximation grad(R(x)) ≈ λ * (x - f(x))
        # A more exact gradient requires differentiating through the denoiser.
        current_lambda = lambda_reg if lambda_reg is not None else self.lambda_reg
        with torch.no_grad():
            kappa_2d_norm = self._preprocess(kappa)
            # Detach input before passing to denoiser
            denoised_norm = self.denoiser(kappa_2d_norm.detach())
            
            #! \nabla_{\kappa} f(\hat{\kappa}) = \frac{\partial f(\hat{\kappa})}{\partial \hat{\kappa}} \frac{\partial \hat{\kappa}}{\partial \kappa}
            
            #! \nabla_{\hat{\kappa}} f(\hat{\kappa}) \approx \hat{\kappa} - f(\hat{\kappa})
            #! Chain rule: \frac{\partial \kappa}{\partial \hat{\kappa}} = \frac{1}{\text{norm\_max} - \text{norm\_min}}
            #! therefore \nabla_{\kappa} f(\kappa) = \frac{1}{\text{norm\_max} - \text{norm\_min}} (\hat{\kappa} - f(\hat{\kappa}))
            grad = (kappa_2d_norm - denoised_norm) / (self.norm_max - self.norm_min)
            grad = grad.flatten().cpu().detach().numpy()
            
        return current_lambda * grad

    def update_lambda(self, lambda_reg: float) -> None:
        """Update the regularization strength."""
        self.lambda_reg = lambda_reg


class CompositeRegularizer(Regularizer):
    """Combines multiple regularizers with individual weights."""

    def __init__(
        self,
        regularizers: List[Regularizer],
        weights: Optional[List[float]] = None,
        global_lambda_reg: float = 1.0, # A global scaling factor
    ):
        self.regularizers = regularizers
        if weights is None:
            self.weights = [1.0] * len(regularizers)
        else:
            if len(weights) != len(regularizers):
                raise ValueError("Number of weights must match number of regularizers")
            self.weights = weights
        self.global_lambda_reg = global_lambda_reg # Store the global lambda

    def __call__(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> float:
        """Calculate the weighted sum of regularization values."""
        current_global_lambda = lambda_reg if lambda_reg is not None else self.global_lambda_reg
        total_value = 0.0
        for reg, weight in zip(self.regularizers, self.weights):
            # Pass the global lambda scaled by the weight to individual regularizers
            # Assumes individual regularizers use the passed lambda_reg override
            total_value += reg(kappa, lambda_reg=current_global_lambda * weight)
        return total_value

    def gradient(
        self, kappa: npt.NDArray[np.float64], lambda_reg: Optional[float] = None
    ) -> npt.NDArray[np.float64]:
        """Calculate the weighted sum of regularization gradients."""
        current_global_lambda = lambda_reg if lambda_reg is not None else self.global_lambda_reg
        total_gradient = np.zeros_like(kappa)
        for reg, weight in zip(self.regularizers, self.weights):
            # Pass the global lambda scaled by the weight
            total_gradient += reg.gradient(kappa, lambda_reg=current_global_lambda * weight)
        return total_gradient

    def update_lambda(self, lambda_reg: float) -> None:
        """Update the global regularization strength."""
        # This updates the global lambda. Individual regularizers will use this
        # scaled by their weight when __call__ or gradient is invoked.
        self.global_lambda_reg = lambda_reg
