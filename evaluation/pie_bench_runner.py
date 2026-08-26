import os
import glob
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from pipelines.flux_adaor_pipeline import FluxAdaOrPipeline

def load_dataset_stratified_pie_bench(mapping_file_path_or_dir: str, images_dir: str, samples_per_category: int = 20) -> List[Dict[str, Any]]:
    """
    Loads PIE-bench dataset from parquet files with embedded image bytes, ensuring a strict stratified split across category folders.
    """
    valid_dataset = []
    
    if os.path.isdir(mapping_file_path_or_dir):
        category_dirs = sorted([
            os.path.join(mapping_file_path_or_dir, d) 
            for d in os.listdir(mapping_file_path_or_dir) 
            if os.path.isdir(os.path.join(mapping_file_path_or_dir, d))
        ])
        
        for cat_dir in category_dirs:
            cat_name = os.path.basename(cat_dir)
            if cat_name.startswith('.'):
                continue
                
            parquet_files = glob.glob(os.path.join(cat_dir, "*.parquet"))
            cat_samples_collected = 0
            
            for p_file in sorted(parquet_files):
                try:
                    df = pd.read_parquet(p_file)
                except Exception as e:
                    print(f"Warning: Failed to read {p_file}: {e}")
                    continue

                for idx, row in df.iterrows():
                    if cat_samples_collected >= samples_per_category:
                        break
                        
                    sample_id = str(row.get("id", f"sample_{idx}"))
                    target_prompt = str(row.get("target_prompt", ""))
                    source_prompt = str(row.get("source_prompt", ""))
                    
                    img_obj = row.get("image", None)
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache", cat_name)
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    
                    if not os.path.exists(img_path):
                        if isinstance(img_obj, dict) and "bytes" in img_obj:
                            img_bytes = img_obj["bytes"]
                        elif isinstance(img_obj, bytes):
                            img_bytes = img_obj
                        else:
                            img_bytes = None
                            
                        if img_bytes is not None:
                            with open(img_path, "wb") as f_img:
                                f_img.write(img_bytes)
                    
                    if os.path.exists(img_path):
                        valid_dataset.append({
                            "id": sample_id,
                            "image_path": img_path,
                            "prompt": target_prompt,
                            "source_prompt": source_prompt,
                            "edit_action": row.get("edit_action", {}),
                            "category": cat_name
                        })
                        cat_samples_collected += 1
                        
                if cat_samples_collected >= samples_per_category:
                    break
            print(f"Category [{cat_name}]: Loaded {cat_samples_collected} samples.")
            
    elif os.path.isfile(mapping_file_path_or_dir):
        # Single parquet file fallback
        df = pd.read_parquet(mapping_file_path_or_dir)
        for idx, row in df.iterrows():
            sample_id = str(row.get("id", f"sample_{idx}"))
            valid_dataset.append({
                "id": sample_id,
                "image_path": os.path.join(images_dir, f"{sample_id}.jpg"),
                "prompt": str(row.get("target_prompt", "")),
                "source_prompt": str(row.get("source_prompt", "")),
                "category": str(row.get("category", "default"))
            })

    return valid_dataset


def run_stratified_pie_bench_eval(
    adaor_pipeline: FluxAdaOrPipeline,
    dataset: List[Dict[str, Any]],
    output_dir: str,
    alpha_steps: List[float],
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42
):
    """
    Executes AdaOr continuous strength sweeps across a stratified PIE-bench dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = getattr(adaor_pipeline.pipeline, "_execution_device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting PIE-Bench Evaluation on {len(dataset)} samples across alpha steps: {alpha_steps}")

    for sample in tqdm(dataset, desc="PIE-Bench AdaOr Evaluation"):
        sample_id = sample["id"]
        category = sample.get("category", "default")
        
        # Clean PIE-bench brackets from prompt if present
        target_prompt = sample["prompt"].replace("[", "").replace("]", "").replace("  ", " ").strip()
        source_prompt = sample.get("source_prompt", "").replace("[", "").replace("]", "").replace("  ", " ").strip()

        sample_out_dir = os.path.join(output_dir, category, sample_id)
        os.makedirs(sample_out_dir, exist_ok=True)

        for alpha in alpha_steps:
            save_path = os.path.join(sample_out_dir, f"alpha_{alpha:.2f}.png")
            if os.path.exists(save_path):
                continue

            generator = torch.Generator(device=device).manual_seed(seed)
            image = adaor_pipeline(
                prompt=target_prompt,
                identity_prompt=source_prompt,
                unconditional_prompt="",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                alpha=alpha,
                generator=generator,
                height=height,
                width=width
            )

            image.save(save_path)

    print(f"🎉 PIE-Bench Stratified Evaluation Complete -> Output saved to {output_dir}")
