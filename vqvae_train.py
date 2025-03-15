from random import sample
import matplotlib.pyplot as plt
from dataset import load_dataset
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import myVQVAE
import metrics
from torchvision.utils import save_image
import os
from kornia.losses import SSIMLoss
# 假设你的数据集存放在 'dataset_path' 路径下，每个类别在不同文件夹内
train_dataset_path      = 'dataset/train'
test_dataset_path       = 'dataset/test'
num_epochs              = 500
initial_learning_rate   = 2e-4
batch_size              = 16
savePath                = 'vae_models'
milestones              = [30, 100]
save_interval           = 20
# 创建一个保存重建图像的文件夹
reconstructed_images_path = os.path.join(savePath, 'reconstructed_images')
if not os.path.exists(reconstructed_images_path):
    os.makedirs(reconstructed_images_path)

# 数据预处理：将图片调整为灰度图，并将大小调整为 256x256
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 加载数据集
train_dataset = load_dataset(
    dataset_root=train_dataset_path, 
    filter_option="healthy", 
    transform=transform
    )
test_dataset  = load_dataset(
    dataset_root=test_dataset_path, 
    filter_option="diseased", 
    transform=transform
    )

# 使用 DataLoader 加载数据
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=1, shuffle=False)

print(f"train data batch num = {len(train_loader)}")
print(f"train test batch num = {len(test_loader)}")

# 定义VQ-VAE模型
def vqvae_loss(recon_x, x, quantized_loss,mse_weight=1.0,ssim_weight=1.0,data_range=2.0):
    # 重建损失
    # MSE损失（保持与原代码兼容性）
    mse_loss = nn.MSELoss(reduction='mean')(recon_x, x)

    # SSIM损失（使用kornia实现）
    ssim_loss = SSIMLoss(
        window_size=11,
        reduction='mean',
        max_val=data_range  # 关键参数！根据输入数据范围调整
    )(recon_x, x)

    # 组合多尺度重建损失
    reconstruction_loss = mse_weight * mse_loss + ssim_weight * ssim_loss

    return reconstruction_loss + quantized_loss

def draw_trainning_process(train_losses, test_losses, psnr_values, ssim_values):
    # 绘制训练损失、测试损失、PSNR 和 SSIM 的可视化图
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # 绘制PSNR曲线
    plt.subplot(2, 2, 2)
    plt.plot(psnr_values, label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')
    plt.legend()

    # 绘制SSIM曲线
    plt.subplot(2, 2, 3)
    plt.plot(ssim_values, label='SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(savePath + '/metrics_plot.png')

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myVQVAE.to(device)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate,weight_decay=1e-5)

scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.5,
    patience=1
)

# 训练过程
train_losses = []
test_losses  = []  # 用于保存每个epoch的测试集损失
psnr_values  = []
ssim_values  = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        images, cond_images, labels = batch["image"], batch["cond_image"], batch["label"]
        images = torch.cat([images, cond_images], dim=0)
        images = images.to(device)
        optimizer.zero_grad()

        # 前向传播
        recon_batch, quantized_loss = model(images)

        # 计算损失
        loss = vqvae_loss(recon_batch, images, quantized_loss)
        loss.backward()

        # 更新模型参数
        optimizer.step()

        train_loss += loss.item()
        if (batch_idx + 1) % (len(train_loader) // 2) == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item() / len(images)}')

    # 每个epoch打印一次平均训练损失
    epoch_loss = train_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'===> Epoch {epoch + 1} Average train loss: {epoch_loss}')

    # 计算并记录测试集损失和重建图像
    model.eval()
    test_loss = 0
    epoch_psnr = []
    epoch_ssim = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, cond_images, labels = batch["image"], batch["cond_image"], batch["label"] 
            images = torch.cat([images, cond_images], dim=0)
            images = images.to(device)

            # 前向传播
            recon_batch, quantized = model(images)
            latents = model.encode(images).to(torch.device('cpu'))

            # 计算损失
            loss = vqvae_loss(recon_batch, images, quantized)
            test_loss += loss.item()

            stacked_images = torch.cat([images, recon_batch], dim=-2).to(torch.device('cpu'))
            
            # 计算 PSNR 和 SSIM
            reconstructed_img = recon_batch.squeeze().to(torch.device('cpu')).numpy()
            original_img = images.squeeze().to(torch.device('cpu')).numpy()

            # # 从 [-1, 1] 转换到 [0, 1]
            original_img = (original_img + 1) / 2
            reconstructed_img = (reconstructed_img + 1) / 2
            # 从 [0, 1] 转换到 [0, 255]
            reconstructed_img = reconstructed_img * 255
            original_img = original_img * 255

            # 确保数据类型为整数
            reconstructed_img = reconstructed_img.astype(np.uint8)
            original_img = original_img.astype(np.uint8)

            # 计算 PSNR 和 SSIM
            psnr_value = metrics.psnr(original_img, reconstructed_img)
            ssim_value = metrics.compute_ssim(original_img, reconstructed_img)

            epoch_psnr.append(psnr_value)
            epoch_ssim.append(ssim_value)
            
            B, C, H, W = images.size()
            flair_img = images[:B//2]
            flair_recon = recon_batch[:B//2]

            # 保存重建图像
            sample_dir = os.path.join(reconstructed_images_path, f"{batch_idx+1}")
            os.makedirs(sample_dir, exist_ok=True)
            save_image(flair_img, os.path.join(sample_dir, f"original.png"))
            save_image(flair_recon, os.path.join(sample_dir, f"reconstruct.png"))
            if (batch_idx+1) % ((batch_idx+1) // 25) == 0:
                save_image(stacked_images, os.path.join(reconstructed_images_path, f'reconstructed_{batch_idx+1}.png'))
                save_image(latents, os.path.join(reconstructed_images_path, f'latents_{batch_idx+1}.png'))

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # 记录 PSNR 和 SSIM
        avg_psnr = np.mean(epoch_psnr)
        avg_ssim = np.mean(epoch_ssim)

        psnr_values.append(avg_psnr)
        ssim_values.append(avg_ssim)

        print(f'===> Epoch {epoch + 1} Average test loss: {test_loss}')
        print(f'===> Epoch {epoch + 1} Average PSNR: {avg_psnr}')
        print(f'===> Epoch {epoch + 1} Average SSIM: {avg_ssim}')

    draw_trainning_process(train_losses, test_losses, psnr_values, ssim_values)
    # 更新学习率
    scheduler.step(epoch_loss)
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()
    print(f'===> Epoch {epoch + 1} Current learning rate: {current_lr}')
    if (epoch + 1) % save_interval == 0:
        save_model_path = os.path.join(savePath, f"vae-{epoch}.pt")
        torch.save(model.state_dict(), save_model_path)
# 保存模型
save_model_path = os.path.join(savePath, f"vae-{epoch}.pt")
torch.save(model.state_dict(), save_model_path)
print(f'Model saved to {save_model_path}')


