import torch

class AdaOrGuidanceProcessor:
    """
    Implements Adaptive-Origin Guidance (AdaOr) for continuous edit scaling.
    Dynamically blends the unconditional baseline with an identity-conditioned 
    baseline to prevent collapse at low/zero strengths, with FLUX norm-preserving 
    rescaling to prevent latent blowout at high alpha.
    """
    def __init__(self, rescale_phi: float = 0.7):
        self.rescale_phi = rescale_phi

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
            guidance_scale: Standard classifier-free guidance (CFG) scale (e.g., 3.5).
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

        # 3. Apply AdaOr Guidance formula
        raw_noise_pred = adaptive_origin + (alpha * guidance_scale) * edit_direction

        # 4. Norm-Preserving Rescaling for FLUX Stability:
        # Rescales feature standard deviation of raw_noise_pred to match target edit prediction std
        # Prevents norm explosion at high alpha values while maintaining guidance_scale=3.5.
        if self.rescale_phi > 0.0:
            std_edit = eps_edit.std(dim=-1, keepdim=True)
            std_raw = raw_noise_pred.std(dim=-1, keepdim=True)
            rescaled_noise_pred = raw_noise_pred * (std_edit / (std_raw + 1e-8))
            noise_pred = self.rescale_phi * rescaled_noise_pred + (1.0 - self.rescale_phi) * raw_noise_pred
        else:
            noise_pred = raw_noise_pred

        return noise_pred