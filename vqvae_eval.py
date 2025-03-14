import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generative.networks.nets import VQVAE
import metrics
import os
from PIL import Image

# 设置设备为CPU
device = torch.device("cpu")

# 数据预处理：将图片调整为灰度图，并将大小调整为 256x256
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 标准化到 [-1, 1]
])

# 加载数据集
dataset_path = 'D:/Desktop/epilepsy/Dataset/PNG_opmd/eval_test'  # 测试数据集路径
test_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载训练好的VQ-VAE模型，并指定将模型加载到CPU
model = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256),  # 确保与下采样层数匹配
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=(
        (2, 4, 1, 1),  # 第一次下采样：256x256 → 64x64
        (2, 4, 1, 1)   # 第二次下采样：64x64 → 16x16
    ),
    upsample_parameters=(
        (2, 4, 1, 1, 0),  # 第一次上采样：16x16 → 64x64
        (2, 4, 1, 1, 0)   # 第二次上采样：64x64 → 256x256
    ),
    num_embeddings=256,
    embedding_dim=4,
    output_act="tanh",
)  # [B, 4, H/16, W/16]
model.load_state_dict(torch.load('.\\premodel\\vqvae_model.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()

# 创建保存图像的父文件夹
save_path = 'D:/Desktop/epilepsy/Dataset/rcSave'
original_save_path = os.path.join(save_path, 'original_images')
reconstructed_save_path = os.path.join(save_path, 'reconstructed_images')
codebook_save_path = os.path.join(save_path, 'codebook_images')  # 新增：保存特征编码图

os.makedirs(original_save_path, exist_ok=True)
os.makedirs(reconstructed_save_path, exist_ok=True)
os.makedirs(codebook_save_path, exist_ok=True)

# 定义保存特征编码图的函数
# 定义保存特征编码图的函数
def save_feature_map(feature_map, save_path, cmap='viridis'):
    """
    保存4通道特征编码图，将其归一化到 [0, 1] 范围
    :param feature_map: 输入的特征编码图 (shape: [C, H, W])
    :param save_path: 保存路径
    :param cmap: 使用的颜色映射，默认为 'viridis'
    """
    # 转换到 numpy 数组
    feature_map = feature_map.cpu().numpy()

    # 动态归一化到 [0, 1]
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

    # 转换为 (H, W, C) 格式
    feature_map_rgb = np.transpose(feature_map, (1, 2, 0))

    # 保存为伪彩色图
    plt.imsave(save_path, feature_map_rgb, cmap=cmap)

# 选择图像进行重建
with torch.no_grad():
    psnr_values = []
    ssim_values = []
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        batch_size, C, H, W = data.shape
        mid_point = H // 2
        mask_upper = torch.zeros(batch_size, C, H, W, device=device)
        mask_upper[:, :, :mid_point, :] = 1
        mask_lower = torch.zeros(batch_size, C, H, W, device=device)
        mask_lower[:, :, mid_point:, :] = 1
        random_choices = torch.rand(batch_size, device=device) > 0.5
        mask = mask_upper
        mask[random_choices] = mask_lower[random_choices]
        data = data * mask
        data[mask == 0] = -1
        # 前向传播
        recon_batch, quantized_loss = model(data)

        # 假设 quantized_loss 是特征编码图
        featuremap_batch = model.encoder(data)
        print(f"Feature map shape: {featuremap_batch.shape}")
        print(f"Reconstructed batch shape: {recon_batch.shape}")
        # 遍历批次中的每张图像
        for img_idx in range(data.size(0)):
            # 获取原图和重建图
            original_img = data[img_idx].squeeze().cpu().numpy()
            reconstructed_img = recon_batch[img_idx].squeeze().cpu().numpy()
            featuremap = featuremap_batch[img_idx]

            # 从 [-1, 1] 转换到 [0, 1]
            original_img = (original_img + 1) / 2
            reconstructed_img = (reconstructed_img + 1) / 2

            # 转换为 uint8 格式 (0-255)
            original_img_uint8 = (original_img * 255).astype(np.uint8)
            reconstructed_img_uint8 = (reconstructed_img * 255).astype(np.uint8)

            # 确保图像维度为 (H, W) 或 (H, W, 3)
            if len(original_img_uint8.shape) == 2:  # 灰度图
                original_img_pil = Image.fromarray(original_img_uint8, mode='L')
                reconstructed_img_pil = Image.fromarray(reconstructed_img_uint8, mode='L')
            else:  # RGB图
                original_img_pil = Image.fromarray(original_img_uint8)
                reconstructed_img_pil = Image.fromarray(reconstructed_img_uint8)

            # 保存原始图像和重建图像
            original_img_pil.save(os.path.join(
                original_save_path,
                f'original_batch{batch_idx}_img{img_idx}.png'
            ))
            reconstructed_img_pil.save(os.path.join(
                reconstructed_save_path,
                f'reconstructed_batch{batch_idx}_img{img_idx}.png'
            ))

            # 保存特征编码图
            code_path = os.path.join(
                codebook_save_path,
                f'codebook_batch{batch_idx}_img{img_idx}.png'
            )
            save_feature_map(featuremap, code_path)
            print(f"Feature map saved to: {code_path}")

            # 计算指标（可选）
            psnr_value = metrics.psnr(original_img_uint8, reconstructed_img_uint8)
            ssim_value = metrics.compute_ssim(original_img_uint8, reconstructed_img_uint8)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            print(f'Batch {batch_idx} Image {img_idx} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}')

        # 如果只需要第一个批次，保留此break
        if batch_idx == 1:
            break

    # 计算平均指标（可选）
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print(f'Average PSNR: {avg_psnr:.4f}')
        print(f'Average SSIM: {avg_ssim:.4f}')