This repository contains code and configuration for the paper titled **SPADE: SEMANTIC-PRESERVING ADAPTIVE DETOXIFICATION OF IMAGES** training and generating detoxified image variants using Sequential ControlNet, as part of our ICLR 2025 submission.

## 🔧 Setup

Clone the diffusers library and install dependencies:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

cd examples/text_to_image
pip install -r requirements_sdxl.txt
cd /content/diffusers/examples/controlnet
```
Run the "final_fine_tune_diffusion_model_controlnet.py" file. 
