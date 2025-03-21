import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import metrics
from model import myVQVAE, args
from dataset import load_dataset
import torch.nn.functional as F

def main():
    # Define paths and parameters
    test_dataset_path = 'dataset/test'
    test_mask_dir = 'dataset/mask_test'  # adjust as needed for your directory structure
    batch_size = 1  # process one sample at a time
    save_dir = os.path.join("vae_models", f"vae-dim{args.embedding_dim}-num{args.num_embeddings}")
    checkpoint_path = os.path.join(save_dir, "vae-final.pt")
    
    # Define test transform (same as in training, note the normalization)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load the test dataset using the updated load_dataset
    test_dataset = load_dataset(
        dataset_root=test_dataset_path,
        mask_dir=test_mask_dir,
        filter_option="both",   # load both healthy and diseased samples
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set up device and load the model
    device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")
    model = myVQVAE.to(device)
    
    # Load the trained model checkpoint (update the path as needed)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Exiting.")
        return
    
    model.eval()
    
    # Lists to accumulate metric values
    psnr_values = []
    ssim_values = []
    
    # Create a folder to store evaluation results
    output_folder = save_dir
    os.makedirs(output_folder, exist_ok=True)
    
    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            # Each batch is a dictionary with keys: image, cond_image, label, mask, filename
            images = batch["image"]          # shape: [1, C, H, W]
            cond_images = batch["cond_image"]  # shape: [1, C, H, W]
            label = batch["label"][0]          # e.g., "normal" or another string
            filename = batch["filename"][0]    # original filename
            mask = batch["mask"][0] if batch["mask"][0] is not None else None
            
            # Concatenate main image and conditional image along the batch dimension
            input_tensor = torch.cat([images, cond_images], dim=0).to(device)  # shape: [2, C, H, W]
            
            # Get model outputs
            recon_tensor, quantized = model(input_tensor)
            latents = model.encode(input_tensor)
            
            # Use only the main image (first half of the concatenated tensor)
            original = input_tensor[0:1]
            reconstruction = recon_tensor[0:1]
            latent_main = latents[0:1]

            B, C, H, W = latent_main.size()
            latent_main = latent_main.view(B*C, 1, H, W)
            
            # Rescale from [-1, 1] to [0, 1] before saving images
            original_norm = (original + 1) / 2
            reconstruction_norm = (reconstruction + 1) / 2
            latent_norm = (latent_main + 1) / 2  # assuming latents are in [-1,1]
            
            # Compute the residual image (absolute difference)
            residual = torch.abs(original_norm - reconstruction_norm)
            residual = (residual - torch.min(residual)) / \
                       (torch.max(residual) - torch.min(residual))
            
            # Compute PSNR and SSIM.
            # Convert tensors to numpy arrays (squeeze removes batch and channel dims).
            orig_np = (original_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
            recon_np = (reconstruction_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
            psnr_val = metrics.psnr(orig_np, recon_np)
            ssim_val = metrics.compute_ssim(orig_np, recon_np)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
            # Create a directory for this sample based on its filename (without extension)
            filename_no_ext = os.path.splitext(filename)[0]
            sample_dir = os.path.join(output_folder, filename_no_ext)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save images (all data is in the range [0, 1])
            save_image(original_norm, os.path.join(sample_dir, "original.png"))
            save_image(reconstruction_norm, os.path.join(sample_dir, "reconstruct.png"))
            save_image(residual, os.path.join(sample_dir, "residual.png"))
            save_image(latent_norm, os.path.join(sample_dir, "latents.png"))
            
            # If the sample is diseased, save the mask as well
            if label.lower() != "normal" and mask is not None:
                save_image(mask, os.path.join(sample_dir, "mask.png"))
            
            print(f"Processed sample {filename} - PSNR: {psnr_val}, SSIM: {ssim_val}")
    
    # Compute and print average metrics over the test set
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Average PSNR over test set: {avg_psnr}")
    print(f"Average SSIM over test set: {avg_ssim}")

if __name__ == '__main__':
    main()
