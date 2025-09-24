
accelerate launch train_controlnet.py \
 --output_dir="dir_name" \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --controlnet_model_name_or_path="your-anonymous-hf-model-id" \
 --dataset_name="your-anonymous-dataset-id" \
 --conditioning_image_column=file_variant1 \
 --image_column=file_variant3 \
 --caption_column=file_variant3_text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./toxic_10_variant3.png" "./toxic_21_variant3.png" \
 --validation_prompt "Prompt 1" "Prompt 2" \
 --train_batch_size=8 \
 --gradient_accumulation_steps=2 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --num_train_epochs=5 \
 --checkpointing_steps=50 \
 --lr_warmup_steps=0 \
 --report_to="wandb" \
 --dataloader_num_workers=8 \
 --allow_tf32 \
 --mixed_precision="fp16" \
 --push_to_hub \
 --hub_model_id="dir_name"


import diffusers
from diffusers import DiffusionPipeline
from diffusers import ControlNetModel, UniPCMultistepScheduler
import os
import pandas as pd
import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

# Paths (update these to match your repo structure)
control_image_dir = "data/control_images"
output_dir = "results/detoxified_outputs"
prompt_csv = "data/prompts.csv"
os.makedirs(output_dir, exist_ok=True)

# Model setup
base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "your-anonymous-hf-model-id"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Load prompts
df = pd.read_csv(prompt_csv)
df['file_variant_2'] = df['file_variant_2'].str.strip()
generator = torch.manual_seed(0)

# Generate images
for idx, row in df.iterrows():
    fname = row['file_variant_2']
    prompt = row['file_variant2_text']
    input_path = os.path.join(control_image_dir, fname)

    if not os.path.exists(input_path):
        print(f"Image not found: {input_path}")
        continue

    try:
        control_image = load_image(input_path)
        output = pipe(
            prompt=prompt,
            num_inference_steps=10,
            generator=generator,
            image=control_image
        ).images[0]

        out_name = os.path.splitext(fname)[0] + "_detoxified.png"
        out_path = os.path.join(output_dir, out_name)
        output.save(out_path)
        print(f"Generated: {out_name}")

    except Exception as e:
        print(f"Failed on {fname}: {e}")


