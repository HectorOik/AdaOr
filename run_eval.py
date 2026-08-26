import os
import yaml
import torch
import argparse
from typing import Optional, List, Union
from diffusers import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from transformers import BitsAndBytesConfig

from pipelines.flux_adaor_pipeline import FluxAdaOrPipeline
from evaluation.pie_bench_runner import load_dataset_stratified_pie_bench, run_stratified_pie_bench_eval

def load_config(config_path: str = "configs/adaor_flux_config.yaml") -> dict:
    if not os.path.isabs(config_path):
        if not os.path.exists(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def parse_alpha_steps(alpha_input: Union[str, List[float]]) -> List[float]:
    if isinstance(alpha_input, list):
        return [float(a) for a in alpha_input]
    if isinstance(alpha_input, str):
        return [float(a.strip()) for a in alpha_input.split(",") if a.strip()]
    return [0.0, 0.25, 0.5, 0.75, 1.0]

def run(
    model_id: Optional[str] = None,
    quantize: Optional[bool] = None,
    config_path: str = "configs/adaor_flux_config.yaml",
    mapping_file: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    alpha_steps: Optional[Union[str, List[float]]] = None,
    samples_per_category: Optional[int] = None,
    stratified: Optional[bool] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed: int = 42,
    prompt: Optional[str] = None,
    identity_prompt: Optional[str] = None,
    unconditional_prompt: str = "",
    overwrite: bool = True
):
    """
    Adaptive entrypoint to execute AdaOr evaluation sweeps on single prompts or stratified PIE-bench datasets.
    """
    cfg = load_config(config_path)
    cfg_model = cfg.get("model", {})
    cfg_eval = cfg.get("evaluation", {})

    # Parameter resolution hierarchy: Direct Argument > Config YAML > Default
    model_id = model_id or cfg_model.get("pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev")
    if quantize is None:
        quantize = cfg_model.get("quantize", False)
    
    output_dir = output_dir or cfg_eval.get("output_dir", "adaor_outputs")
    mapping_file = mapping_file or cfg_eval.get("mapping_file", None)
    images_dir = images_dir or cfg_eval.get("images_dir", "datasets/pie_bench")
    
    if alpha_steps is None:
        alpha_steps = cfg_eval.get("alpha_steps", [0.0, 0.25, 0.5, 0.75, 1.0])
    alpha_steps = parse_alpha_steps(alpha_steps)

    if samples_per_category is None:
        samples_per_category = cfg_eval.get("samples_per_category", 20)
    if stratified is None:
        stratified = cfg_eval.get("stratified", mapping_file is not None and os.path.exists(mapping_file))
    
    num_inference_steps = num_inference_steps or cfg_eval.get("num_inference_steps", 4 if "schnell" in model_id.lower() else 28)
    guidance_scale = guidance_scale if guidance_scale is not None else cfg_eval.get("guidance_scale", 3.5)
    height = height or cfg_eval.get("height", 1024)
    width = width or cfg_eval.get("width", 1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if cfg_model.get("torch_dtype", "bfloat16") == "bfloat16" else torch.float16

    print("==================================================")
    print(f" AdaOr Evaluation Runner")
    print(f" Model ID: {model_id}")
    print(f" Quantization (4-bit NF4): {quantize}")
    print(f" Stratified Mode: {stratified}")
    print(f" Alpha Steps: {alpha_steps}")
    print(f" Device: {device} | Dtype: {dtype}")
    print("==================================================")

    # 1. Model Loading with Optional 4-bit NF4 Quantization
    if quantize:
        print("⚡ Loading Quantized FLUX Transformer (4-bit NF4)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16
        )
        base_pipeline = FluxPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=dtype
        ).to(device)
    else:
        print(f"Loading standard FLUX pipeline from {model_id}...")
        base_pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        ).to(device)

    # Wrap base pipeline with AdaOr processor
    adaor_pipeline = FluxAdaOrPipeline(base_pipeline)

    # 2. Benchmark or Single-Prompt Execution Mode
    if stratified and mapping_file and os.path.exists(mapping_file):
        print(f"Loading stratified PIE-bench dataset from {mapping_file}...")
        dataset = load_dataset_stratified_pie_bench(
            mapping_file_path_or_dir=mapping_file,
            images_dir=images_dir,
            samples_per_category=samples_per_category
        )
        print(f"Loaded {len(dataset)} stratified samples. Running benchmark evaluation...")
        run_stratified_pie_bench_eval(
            adaor_pipeline=adaor_pipeline,
            dataset=dataset,
            output_dir=output_dir,
            alpha_steps=alpha_steps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed
        )
    else:
        # Single sample or demo evaluation sweep
        test_prompt = prompt or "a professional portrait photo of a woman with vibrant red hair smiling"
        identity_prompt_val = identity_prompt or "a professional portrait photo of a woman smiling"

        os.makedirs(output_dir, exist_ok=True)
        print(f"Running single sample AdaOr evaluation sweep across alpha steps: {alpha_steps}")

        for alpha in alpha_steps:
            print(f"-> Generating image for alpha strength: {alpha}")
            generator = torch.Generator(device=device).manual_seed(seed)
            image = adaor_pipeline(
                prompt=test_prompt,
                identity_prompt=identity_prompt_val,
                unconditional_prompt=unconditional_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                alpha=alpha,
                generator=generator,
                height=height,
                width=width
            )
            save_path = os.path.join(output_dir, f"adaor_alpha_{alpha:.2f}.png")
            image.save(save_path)
            print(f"Saved: {save_path}")

    print("🎉 AdaOr evaluation task complete!")


def main():
    parser = argparse.ArgumentParser(description="Continuous Control of Editing Models via Adaptive-Origin Guidance (AdaOr)")
    parser.add_argument("--config", type=str, default="configs/adaor_flux_config.yaml", help="Path to config YAML")
    parser.add_argument("--model_id", type=str, help="Pretrained FLUX model HF path")
    parser.add_argument("--quantize", action="store_true", default=None, help="Enable 4-bit NF4 quantization")
    parser.add_argument("--no_quantize", action="store_false", dest="quantize", help="Disable 4-bit NF4 quantization")
    parser.add_argument("--mapping_file", type=str, help="Path to PIE-bench parquet folder or mapping file")
    parser.add_argument("--images_dir", type=str, help="Directory for PIE-bench extracted images cache")
    parser.add_argument("--output_dir", type=str, help="Directory to save output images")
    parser.add_argument("--alpha_steps", type=str, help="Comma-separated alpha values (e.g. '0.0,0.25,0.5,0.75,1.0')")
    parser.add_argument("--samples_per_category", type=int, help="Samples to select per category for stratified evaluation")
    parser.add_argument("--stratified", action="store_true", default=None, help="Enable stratified PIE-bench sampling")
    parser.add_argument("--no_stratified", action="store_false", dest="stratified", help="Disable stratified sampling")
    parser.add_argument("--num_inference_steps", type=int, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Target edit prompt for single-sample run")
    parser.add_argument("--identity_prompt", type=str, help="Identity source prompt for single-sample run")
    parser.add_argument("--unconditional_prompt", type=str, default="", help="Unconditional prompt")

    args = parser.parse_args()

    run(
        model_id=args.model_id,
        quantize=args.quantize,
        config_path=args.config,
        mapping_file=args.mapping_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        alpha_steps=args.alpha_steps,
        samples_per_category=args.samples_per_category,
        stratified=args.stratified,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
        prompt=args.prompt,
        identity_prompt=args.identity_prompt,
        unconditional_prompt=args.unconditional_prompt
    )

if __name__ == "__main__":
    main()