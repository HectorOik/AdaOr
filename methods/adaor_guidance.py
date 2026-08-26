import torch

class AdaOrGuidanceProcessor:
    """
    Implements Adaptive-Origin Guidance (AdaOr) for continuous edit scaling.
    Dynamically blends the unconditional baseline with an identity-conditioned 
    baseline to prevent collapse at low/zero strengths.
    """
    def __init__(self):
        pass

    def get_adaor_noise_pred(
        self,
        eps_edit: torch.Tensor,
        eps_uncond: torch.Tensor,
        eps_identity: torch.Tensor,
        guidance_scale: float,
        alpha: float
    ) -> torch.Tensor:
        """
        Calculates the final guided noise prediction using the AdaOr formula.
        
        Args:
            eps_edit: Noise prediction conditioned on the target edit prompt.
            eps_uncond: Noise prediction conditioned on the empty/unconditional prompt.
            eps_identity: Noise prediction conditioned on the source identity prompt.
            guidance_scale: Standard classifier-free guidance (CFG) scale (e.g., 7.5).
            alpha: Edit strength scalar parameter ranging from 0.0 to 1.0 (or higher).
            
        Returns:
            The adjusted noise prediction tensor.
        """
        # 1. Compute the Adaptive Origin
        # When alpha = 0.0, origin is eps_identity (preserving source image identity/structure).
        # When alpha = 1.0, origin is eps_uncond (standard CFG origin).
        adaptive_origin = (1.0 - alpha) * eps_identity + alpha * eps_uncond

        # 2. Apply Classifier-Free Guidance relative to the adaptive origin scaled by edit strength alpha
        noise_pred = adaptive_origin + (guidance_scale * alpha) * (eps_edit - adaptive_origin)

        return noise_pred