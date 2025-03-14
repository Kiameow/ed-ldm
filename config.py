# configs.py
from dataclasses import dataclass
from torchvision import transforms
import torch
@dataclass
class TrainConfig:
    dataSize =256
    # 数据配置
    batch_size: int = 4 #不同
    num_workers: int = 1

    # 模型架构配置
    latent_dim: int = 4  # VQVAE潜空间维度
    text_embed_dim: int = 512  # CLIP文本编码维度
    cond_channels: int = 8  # 条件通道数（图像4 + 文本4）
    timesteps: int = 1000  # 扩散时间步数
    latent_shape = (latent_dim,64,64)
    # 训练参数
    scheduler_type: str = "linear"  # 可选 ["linear", "cosine", "sigmoid", "sqrt",""squaredcos_cap_v2]
    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008  # 余弦调度偏移量
    lr: float = 1e-5
    milestones = [400]
    epochs: int = 1000
    save_interval: int = 800  # 模型保存间隔
    device : str= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 路径配置
    train_data_dir: str = "dataset/Train"
    test_data_dir: str = "dataset/Test"
    vqvae_dir: str = "vae_models/vae.pt"
    Diffusion_dir: str = "diffusion_models/diffusion.pt"
    save_best: bool = True  # 是否保存最佳模型
    save_dir: str = "diffusion_models"
    eval_save: str = "results"
    maskPath: str = "dataset/mask_test"
    textCoderPath : str = "text_embedder"

    #classifier free guidance
    cond_drop_prob_img: float = 0  # 图像条件丢弃概率
    cond_drop_prob_text: float = 0  # 文本条件丢弃概率
    guidance_scale_img: float = 2 # 图像引导比例
    guidance_scale_text: float = 1.5 # 文本引导比例
    mean_flair : float = 1.2603
    mean_t1 : float = 1.0541
    std_flair : float = 3.1313
    std_t1 : float = 3.1982
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((dataSize, dataSize)),
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomRotation(10),   # 随机旋转 ±10 度
#        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
