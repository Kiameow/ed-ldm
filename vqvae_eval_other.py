import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import metrics
from model import myVQVAE
import torch.nn.functional as F

def process_image(image_path, mask_path=None):
    # Define test transform (same as in training, note the normalization)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load and process the input image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load and process the mask if provided
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = test_transform(mask).unsqueeze(0)
    
    return image, mask

def main(image_path, mask_path=None):
    checkpoint_path = 'vae_models/vae-final.pt'
    
    # Set up device and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myVQVAE.to(device)
    
    # Load the trained model checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Exiting.")
        return
    
    model.eval()
    
    # Process the input image
    image, mask = process_image(image_path, mask_path)
    image = image.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        recon_tensor, _ = model(image)
        latents = model.encode(image)
    
    # Rescale from [-1, 1] to [0, 1]
    original_norm = (image + 1) / 2
    reconstruction_norm = (recon_tensor + 1) / 2
    
    # Compute the residual image (absolute difference)
    residual = torch.abs(original_norm - reconstruction_norm)
    residual = (residual - torch.min(residual)) / (torch.max(residual) - torch.min(residual))
    
    # Compute PSNR and SSIM
    orig_np = (original_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
    recon_np = (reconstruction_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
    psnr_val = metrics.psnr(orig_np, recon_np)
    ssim_val = metrics.compute_ssim(orig_np, recon_np)
    
    # Create an output folder
    output_folder = "vae_eval_other_results"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the filename without extension
    filename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    sample_dir = os.path.join(output_folder, filename_no_ext)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save images
    save_image(original_norm, os.path.join(sample_dir, "original.png"))
    save_image(reconstruction_norm, os.path.join(sample_dir, "reconstruct.png"))
    save_image(residual, os.path.join(sample_dir, "residual.png"))
    
    # Save mask if provided
    if mask is not None:
        save_image(mask, os.path.join(sample_dir, "mask.png"))
    
    print(f"Processed image {image_path} - PSNR: {psnr_val}, SSIM: {ssim_val}")
    print(f"Results saved in {sample_dir}")

if __name__ == '__main__':
    image_path = "dataset/brain/flower.png"  # Update this to your test image
    mask_path = None  # Optional: Update if you have a mask
    main(image_path, mask_path)
