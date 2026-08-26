import torch
from typing import Dict, Any, Optional
from diffusers import FluxPipeline
from methods.adaor_guidance import AdaOrGuidanceProcessor

class FluxAdaOrPipeline:
    """
    Wrapper around the Hugging Face FLUX pipeline to inject AdaOr 
    guidance calculations at every denoising step.
    """
    def __init__(self, pipeline: FluxPipeline):
        self.pipeline = pipeline
        self.processor = AdaOrGuidanceProcessor()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        identity_prompt: str,
        unconditional_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        alpha: float = 1.0,
        generator: Optional[torch.Generator] = None,
        height: int = 1024,
        width: int = 1024
    ):
        """
        Executes the FLUX denoising loop with adaptive-origin guidance modification.
        """
        device = getattr(self.pipeline, "_execution_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = self.pipeline.transformer.dtype

        # 1. Prepare text embeddings for Edit, Unconditional, and Identity prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, device=device
        )
        uncond_embeds, pooled_uncond_embeds, uncond_text_ids = self.pipeline.encode_prompt(
            prompt=unconditional_prompt, prompt_2=None, device=device
        )
        identity_embeds, pooled_identity_embeds, identity_text_ids = self.pipeline.encode_prompt(
            prompt=identity_prompt, prompt_2=None, device=device
        )

        # 2. Initialize latent representations using standard scheduler logic
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=None
        )

        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        # Handle FLUX transformer internal guidance embedding
        guidance = None
        adaor_step_guidance = guidance_scale
        if getattr(self.pipeline.transformer.config, "guidance_embeds", False):
            # Pass guidance_scale (3.5) into FLUX transformer internal guidance embedding
            guidance = torch.full((latents.shape[0],), guidance_scale, device=device, dtype=dtype)
            # Since guidance is embedded into the transformer predictions, use 1.0 in AdaOr math to avoid double scaling
            adaor_step_guidance = 1.0

        # 3. Custom Denoising Loop executing 3 forward passes per step
        for i, t in enumerate(timesteps):
            # Scale timestep for FLUX transformer input format if needed
            timestep = t.expand(latents.shape[0]).to(dtype)
            if timestep.max() > 10.0:
                timestep_input = timestep / 1000.0
            else:
                timestep_input = timestep

            # Helper for forward pass
            def forward_pass(encoder_hidden_states, pooled_projections, txt_ids):
                kwargs = {
                    "hidden_states": latents,
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                    "timestep": timestep_input,
                    "img_ids": latent_image_ids,
                    "txt_ids": txt_ids,
                    "return_dict": False
                }
                if guidance is not None:
                    kwargs["guidance"] = guidance
                return self.pipeline.transformer(**kwargs)[0]

            # --- Forward Pass 1: Target Edit Condition ---
            eps_edit = forward_pass(prompt_embeds, pooled_prompt_embeds, text_ids)

            # --- Forward Pass 2: Unconditional prediction ---
            eps_uncond = forward_pass(uncond_embeds, pooled_uncond_embeds, uncond_text_ids)

            # --- Forward Pass 3: Identity prediction ---
            eps_identity = forward_pass(identity_embeds, pooled_identity_embeds, identity_text_ids)

            # 4. Apply AdaOr Math Processor
            noise_pred = self.processor.get_adaor_noise_pred(
                eps_edit=eps_edit,
                eps_uncond=eps_uncond,
                eps_identity=eps_identity,
                guidance_scale=adaor_step_guidance,
                alpha=alpha
            )

            # 5. Step scheduler forward
            latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 6. Decode latents back into final image space
        latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
        latents = (latents / self.pipeline.vae.config.scaling_factor) + getattr(self.pipeline.vae.config, "shift_factor", 0.0)
        image = self.pipeline.vae.decode(latents, return_dict=False)[0]
        image = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

        return image