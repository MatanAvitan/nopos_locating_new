import torch
checkpoint = torch.load("nanoGPT/out-2layer-mechanism-bos80/R0/final_ckpt.pt", map_location="cpu", weights_only=False)
if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
    print(f"Config: {checkpoint['config']}")
if 'wandb_id' in checkpoint:
    print(f"WandB ID: {checkpoint['wandb_id']}")
# Check for common keys that might contain the run URL or project name
for key in ['run_id', 'wandb_run_id', 'project']:
    if key in checkpoint:
        print(f"{key}: {checkpoint[key]}")
