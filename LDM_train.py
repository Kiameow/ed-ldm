# train.py
import os
import logging
import sys
from datetime import datetime
from config import TrainConfig
from model import LatentDiffusion, myVQVAE
from dataset import get_dataloaders
import torch
from matplotlib import pyplot as plt
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch_ema import ExponentialMovingAverage
import time
import metrics

# ==== 新增日志配置 ====
def setup_logger(save_dir):
    """配置日志记录器"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件日志
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def train():
    try:
        # 初始化配置
        print(torch.backends.mps.is_available())
        config = TrainConfig()
        os.makedirs(config.save_dir, exist_ok=True)

        logger = setup_logger(config.save_dir)  # 初始化日志

        logger.info("==== 开始训练 ====")
        logger.info(f"设备: {config.device}")
        logger.info(f"保存目录: {config.save_dir}")

        vqvae = myVQVAE.to(config.device)
        vqvae.load_state_dict(torch.load(config.vqvae_dir, map_location=config.device))
        vqvae.eval()
        # 初始化扩散模型
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.scheduler_type
        )

        model = LatentDiffusion(
            config,
            noise_scheduler,
            cond_drop_prob_img=config.cond_drop_prob_img,
            cond_drop_prob_text=config.cond_drop_prob_text
        ).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        # 获取数据加载器
        train_loader, test_loader = get_dataloaders(config)  # 修改后的数据加载器
        print(config.device)
        logger.info(f"训练集批次: {len(train_loader)}, 测试集批次: {len(test_loader)}")
        # 学习率调度器（使用训练集长度计算）
        optimizer_scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

        ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

        # 训练记录
        train_losses = []
        test_losses = []
        psnr_values = []
        ssim_values = []
        best_loss = float('inf')

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数量: {trainable_params}")
        text_to_label = {
            'normal': 1 ,
            'Epilepsy (FCD)': 2
        }
        # 记录训练开始时间
        start_time = time.time()
        # 训练循环
        print("epochs is", config.epochs)
        for epoch in range(config.epochs):
            try:
                # ================= 训练阶段 =================
                logger.info(f"\n==== Epoch {epoch + 1}/{config.epochs} ====")
                model.train()
                epoch_train_loss = []

                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # 转移数据到GPU
                        img = batch['image'].to(config.device)
                        cond_img = batch['cond_image'].to(config.device)
                        texts = batch['text']
                        text_cond = torch.tensor([text_to_label[s] for s in texts]).to(config.device).long()
                        # 显存监控
                        if batch_idx % 10 == 0:
                            logger.debug(
                                f"Batch {batch_idx} - GPU显存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")

                        # 使用VQVAE编码图像
                        with torch.no_grad():
                            z = (vqvae.encode_stage_2_inputs(img) - config.mean_flair) / config.std_flair   # 获取量化后的潜变量
                            cond_z = (vqvae.encode_stage_2_inputs(cond_img) - config.mean_t1) /config.std_t1
                            # print(z.shape, cond_z.shape)
                            # print(text_cond.shape)
                        # 前向传播
                        loss = model(z, cond_z,text_cond)

                        # 反向传播
                        
                        loss.backward()
#                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        ema.update()

                        epoch_train_loss.append(loss.item())
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} 处理失败: {str(e)}")
                        logger.error("跳过该批次，继续训练...")
                        continue  # 跳过当前batch

                # 计算平均训练损失
                # 更新学习率
                optimizer_scheduler.step()

                avg_train_loss = sum(epoch_train_loss) / len(train_loader)
                train_losses.append(avg_train_loss)
                logger.info(f"训练损失: {avg_train_loss:.4f}")
                # ================= 验证阶段 =================
                model.eval()
                epoch_test_loss = []
                psnr_sum = 0.0
                ssim_sum = 0.0
                test_metrics = (epoch + 1) % 50 == 0
                test_loss_compute = (epoch + 1) % 10 == 0

                with torch.no_grad():
                    if test_loss_compute or test_metrics:
                        for i, batch in enumerate(test_loader):
                            img = batch['image'].to(config.device)
                            cond_img = batch['cond_image'].to(config.device)
                            z = (vqvae.encode_stage_2_inputs(img) - config.mean_flair) / config.std_flair  # 获取量化后的潜变量
                            cond_z = (vqvae.encode_stage_2_inputs(cond_img) - config.mean_t1) /config.std_t1

                            texts = batch['text']
                            text_cond = torch.tensor([text_to_label[s] for s in texts]).to(config.device).long()
                            
                            # 计算验证损失
                            loss = model(z, cond_z,text_cond)
                            epoch_test_loss.append(loss.item())

                            # 生成图像并计算PSNR和SSIM
                            if test_metrics and i < 5:
                                if epoch >= config.epochs:  # 只在第251个epoch及之后计算指标
                                    generated_z = model.sample(cond_z,guidance_scale_img=3.0,guidance_scale_text=5.0)
                                    generated_img = vqvae.decode(generated_z).detach()
                                    generated_img = (generated_img + 1.0) * 127.5

                                    original_img = (img + 1.0) * 127.5

                                    psnr_val = metrics.torch_psnr(generated_img, original_img).item()
                                    ssim_val = metrics.torch_ssim(generated_img, original_img).item()
                                    psnr_sum += psnr_val
                                    ssim_sum += ssim_val

                 # 计算平均验证损失
                if test_loss_compute:
                    avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
                    test_losses.append(avg_test_loss)
                    logger.info(f"测试损失: {avg_test_loss:.4f}")
                else:
                    avg_test_loss = None  # 表示未计算

                # 处理PSNR和SSIM
                if test_metrics and epoch >= 250:
                    avg_psnr = psnr_sum / 5
                    avg_ssim = ssim_sum / 5
                    psnr_values.append(avg_psnr)
                    ssim_values.append(avg_ssim)
                    logger.info(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

                # ================= 保存和日志 =================
                print_msg = f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}"
                if avg_test_loss is not None:
                    print_msg += f" | Test Loss: {avg_test_loss:.4f}"
                if test_metrics and epoch >= 250:
                    print_msg += f" | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}"
                print(print_msg)
                # 保存最佳模型
                if avg_test_loss is not None and avg_test_loss < best_loss:

                    best_loss = avg_test_loss
                    torch.save(model.state_dict(), f"{config.save_dir}/best_model.pt")
                    logger.info(f"发现最佳模型 (损失: {avg_test_loss:.4f})")

                # 定期保存检查点
                if (epoch + 1) % config.save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'ema': ema.state_dict()
                    }, f"{config.save_dir}/checkpoint_{epoch + 1}.pt")

                
                # 绘制损失和指标曲线
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(train_losses, label='Train Loss')
                plt.plot([i for i in range(len(test_losses)) if (i + 1) % 10 == 0],
                         [test_losses[i] for i in range(len(test_losses)) if (i + 1) % 10 == 0],
                         label='Test Loss (every 10 epochs)')
                plt.title('Loss Curve')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend()

                plt.subplot(1, 2, 2)
                metrics_epochs = [epoch + 1 for epoch in range(len(psnr_values)) if (epoch + 1) % 50 == 0]
                if len(metrics_epochs) > 0:
                    plt.plot(metrics_epochs, psnr_values, label='PSNR (every 50 epochs)')
                    plt.plot(metrics_epochs, ssim_values, label='SSIM (every 50 epochs)')
                plt.title('Metrics Curve')
                plt.ylabel('Metrics')
                plt.xlabel('Epoch')
                plt.legend()

                plt.tight_layout()
                plt.savefig(f"{config.save_dir}/metrics_loss.png")
                plt.close()
            except Exception as e:
                logger.error(f"Epoch {epoch} 执行失败: {str(e)}")
                logger.error("尝试保存崩溃检查点...")
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, f"{config.save_dir}/crash_checkpoint.pt")
                raise  # 重新抛出异常以保留堆栈

        # 训练结束保存最终模型
        print("训练完成")
        # 记录训练结束时间
        end_time = time.time()

        # 计算总运行时间
        total_time = (end_time - start_time)/3600
        logger.info(f"训练总运行时间: {total_time} hours")
        torch.save(model.state_dict(), f"{config.save_dir}/final_model.pt")
    except Exception as e:
        logger.critical(f"训练致命错误: {str(e)}", exc_info=True)  # 记录完整堆栈
        sys.exit(1)  # 确保进程退出


if __name__ == "__main__":
    train()
