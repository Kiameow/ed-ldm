
import torch
import torch.nn as nn
from config import TrainConfig
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from modules.unet import UNetModel
from modules.udit_models import U_DiT
from generative.networks.nets import VQVAE

myVQVAE = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 512),  # Adding an extra channel for the third downsampling layer
    num_res_channels=512,
    num_res_layers=2,
    downsample_parameters=(
        (2, 4, 1, 1),  # First downsampling: 256x256 → 128x128
        (2, 4, 1, 1),  # Second downsampling: 128x128 → 64x64
        (2, 4, 1, 1),  # Third downsampling: 64x64 → 32x32
    ),
    upsample_parameters=(
        (2, 4, 1, 1, 0),  # First upsampling: 32x32 → 64x64
        (2, 4, 1, 1, 0),  # Second upsampling: 64x64 → 128x128
        (2, 4, 1, 1, 0),  # Third upsampling: 128x128 → 256x256
    ),
    num_embeddings=4,
    embedding_dim=1
)

# diffusion.py
class ConditionalUNet(nn.Module):
    """基于第三方库的条件UNet适配器"""

    def __init__(self,TrainConfig):
        super().__init__()
        # 计算总输入通道数（潜空间+条件图像+条件文本）
        in_channels = TrainConfig.latent_dim *2
        # 初始化第三方UNet
        self.unet = U_DiT(
            input_size=TrainConfig.dataSize / 4,
            down_factor=2,
            in_channels=8,
            hidden_size=96,
            depth=[2, 5, 8, 5, 2],
            num_heads=4,
            mlp_ratio=2,
            class_dropout_prob=0,
            num_classes=3,
            learn_sigma=True,
            rep=1,
            ffn_type='rep',
            attn_type='v2',
            posemb_type='rope2d',
            downsampler='dwconv5',
            down_shortcut=1
        )


    def forward(self, x, t,cond_text):
        """
        参数说明：
        x: 噪声潜变量 [B, C, H, W]
        t: 时间步 [B]
        cond_text: 文本条件
        """
        # 拼接潜变量和条件特征
        pre_x = self.unet(x, t, cond_text)
        return pre_x


class LatentDiffusion(nn.Module):
    """潜空间扩散模型"""

    def __init__(self, TrainConfig, noise_scheduler, cond_drop_prob_img=0.1, cond_drop_prob_text=0.1):
        super().__init__()
        self.config = TrainConfig
        self.unet = ConditionalUNet(TrainConfig)
        self.betas = torch.linspace(1e-4, 0.02, TrainConfig.timesteps)
        self.alphas = 1 - self.betas
        self.noise_scheduler = noise_scheduler
        self.cond_drop_prob_img = cond_drop_prob_img
        self.cond_drop_prob_text = cond_drop_prob_text


#    def forward2(self, z, cond_img, text_cond):
#        """
#        前向传播流程
#        :param x: 原始输入图像 [B, 1, H, W]
#        :param cond_img: 条件图像 [B, N]
#        :param texts: 文本描述列表 [B]
#        """
#        # 独立生成图像和文本的掩码
#        mask_img = torch.rand(z.size(0), device=z.device) > self.cond_drop_prob_img
#        mask_text = torch.rand(z.size(0), device=z.device) > self.cond_drop_prob_text
#
#        cond_img = cond_img * mask_img[:, None, None, None]
#        text_cond = text_cond * mask_text
#
#        device = z.device
#        b = z.shape[0]
#        t = torch.randint(0, self.config.timesteps, (b,), device=device).long()
#
#        # 添加噪声
#        noise = torch.randn_like(z).to(device)
#        # alpha_t = self.alphas[t].view(-1, 1, 1, 1).to(device)
#        # 使用 noise_scheduler 添加噪声（关键修改）
#        z_noisy = self.noise_scheduler.add_noise(z, noise, t).to(device)
#
#        # 3. 拼接图像
#        combined_imgs = torch.cat([z_noisy, cond_img], dim=1)  # [B, 8, 64, 64]
#
#        # 预测噪声
#        pred_noise = self.unet(combined_imgs, t)
#
#        return F.mse_loss(pred_noise[:, :4].float(), noise.float())
    
    def forward(self, z, cond_img,cond_text):
        """
        前向传播流程
        :param x: 原始输入图像 [B, 1, H, W]
        :param cond_img: 条件图像 [B, N]
        """
        # 独立生成图像和文本的掩码
        mask_text = torch.rand(z.size(0), device=z.device) > self.cond_drop_prob_img

        cond_text = cond_text * mask_text

        device = z.device
        b = z.shape[0]
        t = torch.randint(0, self.config.timesteps, (b,), device=device).long()

        # 添加噪声
        noise = torch.randn_like(z).to(device)
        # alpha_t = self.alphas[t].view(-1, 1, 1, 1).to(device)
        # 使用 noise_scheduler 添加噪声（关键修改）
        z_noisy = self.noise_scheduler.add_noise(z, noise, t).to(device)

        # 3. 拼接图像
        combined_imgs = torch.cat([z_noisy, cond_img], dim=1)  # [B, 8, 64, 64]
        # 预测噪声
        pred_noise = self.unet(combined_imgs, t,cond_text)

        return F.mse_loss(pred_noise[:, :4].float(), noise.float())

    def sample(self, cond_img, cond_text, num_steps=None, guidance_scale_img=3, guidance_scale_text=5):
        self.eval()
        device = next(self.parameters()).device
        num_steps = num_steps or self.config.timesteps

        z_t = torch.randn_like(cond_img, device=device)
        batch_size = cond_img.size(0)

        self.noise_scheduler.set_timesteps(num_steps, device=device)

        # 去噪
        for t in self.noise_scheduler.timesteps:
            # 拼接当前潜变量与条件图像
            combined_input = torch.cat([z_t, cond_img], dim=1)

            # 合并所有条件输入
            model_input = combined_input
            t_batch = torch.tensor([t], device=device).repeat(batch_size)

            # 准备文本条件和无条件输入
            unconditional_text = torch.zeros_like(cond_text)  # 或者使用随机噪声

            # 批量前向预测
            with torch.no_grad():
                # 条件文本预测
                noise_pred_text = self.unet(model_input, t_batch, cond_text)
                # 无条件文本预测
                noise_pred_uncond_text = self.unet(model_input, t_batch, unconditional_text)

                noise_pred = noise_pred_uncond_text + guidance_scale_text * (noise_pred_text - noise_pred_uncond_text)

            noise_pred = noise_pred[:, :4].float()
            z_t = self.noise_scheduler.step(
                noise_pred,
                t,
                z_t
            ).prev_sample

        return z_t
if __name__ == "__main__":
    print("Number of model parameters:", sum([p.numel() for p in myVQVAE.parameters()]))

