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
            guidance_scale: Guidance scale (applied if not already embedded in model outputs).
            alpha: Edit strength scalar parameter ranging from 0.0 to 1.0.
            
        Returns:
            The adjusted noise prediction tensor.
        """
        # 1. Compute the Adaptive Origin:
        # When alpha = 0.0, origin is eps_identity (preserving source structure).
        # When alpha = 1.0, origin is eps_uncond (standard CFG origin).
        adaptive_origin = (1.0 - alpha) * eps_identity + alpha * eps_uncond

        # 2. Compute direction vector from origin to edit condition
        edit_direction = eps_edit - adaptive_origin

        # 3. Apply AdaOr Guidance formula:
        # Interpolates noise prediction along the edit direction weighted by alpha
        noise_pred = adaptive_origin + (alpha * guidance_scale) * edit_direction

        return noise_pred