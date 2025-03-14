import torch
import numpy as np
from dataset_eval import get_dataloaders, MultiConditionDataset
from config import TrainConfig
from model import LatentDiffusion, myVQVAE
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from diffusers import DDPMScheduler
import metrics as Metrics
from torchvision.io import read_image
from torchvision.utils import save_image
config = TrainConfig()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

def save_results_to_txt(results, file_path):
    with open(file_path, 'w') as f:
        # 写入 PSNR 和 SSIM
        f.write(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f}\n")
        f.write(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}\n\n")

        # 写入阈值对应指标
        f.write("Threshold Metrics:\n")
        f.write("Threshold\tDice\tIoU\tPrecision\tRecall\tAccuracy\tF1\n")
        for t in sorted(results['threshold_metrics'].keys()):
            metrics = results['threshold_metrics'][t]
            f.write(
                f"{t:.2f}\t"
                f"{metrics['dice']['mean']:.4f}±{metrics['dice']['std']:.4f}\t"
                f"{metrics['iou']['mean']:.4f}±{metrics['iou']['std']:.4f}\t"
                f"{metrics['precision']['mean']:.4f}±{metrics['precision']['std']:.4f}\t"
                f"{metrics['recall']['mean']:.4f}±{metrics['recall']['std']:.4f}\t"
                f"{metrics['acc']['mean']:.4f}±{metrics['acc']['std']:.4f}\t"
                f"{metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f}\n"
            )

# 修改数据集类以保留文件名信息
class EnhancedDataset(MultiConditionDataset):
    def _load_data_paths(self):
        with open(self.text_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",", 1) for line in f.readlines()]

        img_paths = []
        filenames = []
        cond_img_paths = []
        texts = []

        for filename, text in lines:
            img_path = os.path.join(self.img_dir, filename)
            cond_img_path = os.path.join(self.cond_img_dir, filename)

            if os.path.exists(img_path) and os.path.exists(cond_img_path):
                img_paths.append(img_path)
                filenames.append(filename) # 记录文件名
                cond_img_paths.append(cond_img_path)
                texts.append(text)

        return img_paths, cond_img_paths, texts, filenames
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data["filename"] = os.path.basename(self.img_paths[idx]) # 添加文件名
        return data


# 加载模型（保持原样）
def load_model2():
    vqvae = myVQVAE.to(device)
    vqvae.load_state_dict(torch.load(config.vqvae_dir, map_location=device))
    vqvae.eval()
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.scheduler_type
    )
    model = LatentDiffusion(config, noise_scheduler).to(device)
    model.load_state_dict(torch.load(config.Diffusion_dir,map_location=device))
    return vqvae, model

def load_model():
    # 加载 VQVAE 模型
    vqvae = myVQVAE.to(device)
    vqvae.load_state_dict(torch.load(config.vqvae_dir, map_location=device))
    vqvae.eval()

    # 初始化噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.scheduler_type
    )

    # 初始化 LatentDiffusion 模型
    model = LatentDiffusion(config, noise_scheduler).to(device)

    # 加载 Diffusion 模型的检查点
    checkpoint = torch.load(config.Diffusion_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])

    return vqvae, model


def match_histograms_gpu(source, target, channel_axis=1):
    """
    GPU版本的直方图匹配
    Args:
        source: 待匹配图像 (Tensor, shape=[B,C,H,W])
        target: 目标图像 (Tensor, shape=[B,C,H,W])
        channel_axis: 通道轴 (默认为1)
    Returns:
        matched: 匹配后的图像 (Tensor)
    """
    matched = []
    for s, t in zip(source, target):
        s_flat = s.reshape(s.shape[channel_axis], -1)  # [C, H*W]
        t_flat = t.reshape(t.shape[channel_axis], -1)

        s_sorted = torch.sort(s_flat, dim=1)[0]
        t_sorted = torch.sort(t_flat, dim=1)[0]

        # 计算映射关系
        quantiles = torch.linspace(0, 1, steps=s_flat.shape[1], device=s.device)
        s_quantiles = torch.quantile(s_sorted, quantiles, dim=1)
        t_quantiles = torch.quantile(t_sorted, quantiles, dim=1)

        # 插值匹配
        matched_flat = torch.zeros_like(s_flat)
        for c in range(s_flat.shape[0]):
            interp = torch_interp(s_flat[c], s_quantiles[c], t_quantiles[c])
            matched_flat[c] = interp

        matched.append(matched_flat.reshape(s.shape))

    return torch.stack(matched)
# 指标计算函数
def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().bool()
    gt = gt_mask.flatten().bool()

    tp = torch.sum(pred & gt).float()
    fp = torch.sum(pred & ~gt).float()
    fn = torch.sum(~pred & gt).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (pred == gt).float().mean()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return dice, iou, precision, recall, acc, f1


def torch_interp(x, xp, fp):
    """
    自定义PyTorch一维线性插值函数，模拟numpy.interp
    参数：
        x: 待插值的点 (Tensor)
        xp: 已知的x坐标（必须单调递增）(Tensor)
        fp: 已知的y坐标 (Tensor)
    返回：
        插值后的值 (Tensor)
    """
    # 处理边界条件，确保xp是单调递增的
    xp, indices_sort = torch.sort(xp)
    fp = fp[indices_sort]

    # 查找插入位置
    indices = torch.searchsorted(xp, x, right=True)

    # 限制索引在有效范围内
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # 获取相邻的xp和fp值
    x_low = xp[indices - 1]
    x_high = xp[indices]
    f_low = fp[indices - 1]
    f_high = fp[indices]

    # 计算插值权重
    delta = x_high - x_low
    delta = torch.where(delta == 0, 1e-8, delta)  # 避免除以零
    weight = (x - x_low) / delta

    # 线性插值
    interp_val = f_low + weight * (f_high - f_low)

    # 处理x超出xp范围的情况
    interp_val = torch.where(x < xp[0], fp[0], interp_val)
    interp_val = torch.where(x > xp[-1], fp[-1], interp_val)

    return interp_val


def match_histograms_gpu(source, target, channel_axis=1):
    matched = []
    for s, t in zip(source, target):
        s_flat = s.reshape(s.shape[channel_axis], -1)  # [C, H*W]
        t_flat = t.reshape(t.shape[channel_axis], -1)

        s_sorted = torch.sort(s_flat, dim=1)[0]
        t_sorted = torch.sort(t_flat, dim=1)[0]

        quantiles = torch.linspace(0, 1, steps=s_flat.shape[1], device=s.device)
        s_quantiles = torch.quantile(s_sorted, quantiles, dim=1)
        t_quantiles = torch.quantile(t_sorted, quantiles, dim=1)

        matched_flat = torch.zeros_like(s_flat)
        for c in range(s_flat.shape[0]):
            # 确保分位数单调递增
            s_q, _ = torch.sort(s_quantiles[c])
            t_q, _ = torch.sort(t_quantiles[c])
            # 使用自定义插值
            interp = torch_interp(s_flat[c], s_q, t_q)
            matched_flat[c] = interp

        matched.append(matched_flat.reshape(s.shape))
    return torch.stack(matched)

def evaluate():
    vqvae, model = load_model2()
    test_dataset = EnhancedDataset(
        root_dir=config.test_data_dir,
        config=config,
        transform=config.transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    thresholds = np.linspace(0, 1, 11)
    metrics = {t: {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'acc': [], 'f1': []}
               for t in thresholds}
    all_psnr = []
    all_ssim = []

    # 定义mask根目录
    mask_root = config.maskPath
    text_to_label = {
        'normal': 1,
        'Epilepsy (FCD)': 2
    }
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        if batch_idx == 0:
            filenames = batch['filename']
            mask_paths = [os.path.join(mask_root, fname) for fname in filenames]
            gt_masks = []
            for mp in mask_paths:
                mask = read_image(mp).to(device).float() / 255.0

                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0),
                    size=(256, 256),
                    mode='nearest'
                ).squeeze(0)


                gt_masks.append(mask)

            gt_masks = torch.stack(gt_masks, dim=0)
            gt_masks = gt_masks.squeeze(1)

            img = batch['image'].to(device)
            cond_img = batch['cond_image'].to(device)
            texts = batch['text']
            text_cond = torch.tensor([text_to_label[s] for s in texts]).to(config.device)
            with torch.no_grad():
                cond_z = (vqvae.encode_stage_2_inputs(cond_img) - config.mean_t1) / config.std_t1

                # 编码器提取特征图
                encoded_features = vqvae.encode_stage_2_inputs(img).detach()
                encoded_features = encoded_features.to(device)
                encoded_features = encoded_features.clamp(-2.0, 2.0)
                encoded_features = (encoded_features + 2.0) / 4.0 #(0,1)
                input_feature_maps_path = os.path.join(config.eval_save, "input_feature_maps")
                os.makedirs(input_feature_maps_path, exist_ok=True)

                # 保存输入特征图的每个通道
                for i, filename in enumerate(filenames):
                    input_feature = encoded_features[i]
                    for channel in range(input_feature.shape[0]):
                        channel_data = input_feature[channel]
                        channel_map = channel_data.unsqueeze(0)
                        channel_name = f"input_feature_channel_{channel}_{filename}"
                        channel_path = os.path.join(input_feature_maps_path, channel_name)
                        save_image(channel_map, channel_path)
                reconstructed = model.sample(cond_img=cond_z,
                                             cond_text=text_cond,
                                             num_steps = config.num_steps,
                                             guidance_scale_img=config.guidance_scale_img,
                                             guidance_scale_text=config.guidance_scale_text
                                             )
                # 保存特征图（归一化到 [0, 255]）
                feature_maps_path = os.path.join(config.eval_save, "feature_maps")
                os.makedirs(feature_maps_path, exist_ok=True)

                for i, filename in enumerate(filenames):
                    feature_map = reconstructed[i] #(-1,1)
                    feature_map = feature_map * 0.5 + 0.5 #(0.0, 1.0)
#                    feature_map = torch.clamp(feature_map, 0.0, 1.0)
                    # 遍历每个通道并单独保存
                    for channel in range(feature_map.shape[0]):
                        channel_map = feature_map[channel]
                        # 将通道数据调整为 (1, h, w)
                        channel_map = channel_map.unsqueeze(0)

                        channel_name = f"feature_channel_{channel}_{filename}"
                        channel_path = os.path.join(feature_maps_path, channel_name)
                        save_image(channel_map, channel_path)
                reconstructed = reconstructed*config.std_flair +config.mean_flair
                reconstructed = vqvae.decode_stage_2_outputs(reconstructed)
#                reconstructed = torch.clamp(reconstructed, -1.0, 1.0)

            # 直方图匹配
            original = img.to(device)
            # matched = match_histograms_gpu(reconstructed, original, channel_axis=1)
            matched = reconstructed
            # original_uint8 = ((original + 1) * 127.5).clamp(0, 255).byte()
            # recon_uint8 = ((matched + 1) * 127.5).clamp(0, 255).byte()
#            original_uint8 = ((original + 1) / 2).clamp(0, 1)
#            recon_uint8 = ((matched + 1) / 2).clamp(0, 1)
            original_uint8 = ((original + 1) / 2)
            recon_uint8 = ((matched + 1) / 2)

            #重建图像保存

            reconstructed_path = config.eval_save
            for i, filename in enumerate(filenames):
                save_name = f"reconstructed_{filename}"
                save_image(recon_uint8[i], os.path.join(reconstructed_path, save_name))
            # 计算指标
#            original_uint8 = (original_uint8 * 255).clamp(0, 255).byte()
#            recon_uint8 = (recon_uint8 * 255).clamp(0, 255).byte()
            original_uint8 = (original_uint8 * 255).byte()
            recon_uint8 = (recon_uint8 * 255).byte()

            all_psnr.append(Metrics.torch_psnr(original_uint8, recon_uint8).item())
            all_ssim.append(Metrics.torch_ssim(original_uint8, recon_uint8).item())

            # 生成差异图
            diff_map = (recon_uint8.float() - original_uint8.float()).abs() / 255.0  # 归一化到[0,1]
            diff_map = diff_map.mean(dim=1)
#            print(diff_map.shape)
            # 遍历阈值计算指标
            for t in thresholds:
                pred_masks = (diff_map > t).float()
                threshold_dir = os.path.join(reconstructed_path, f"threshold_{t:.2f}")
                os.makedirs(threshold_dir, exist_ok=True)

                for i, (filename, pred, gt) in enumerate(zip(filenames, pred_masks, gt_masks)):
                    save_name = f"pred_mask_{filename}"

                    dice, iou, p, r, acc, f1 = calculate_metrics(pred, gt)
                    metrics[t]['dice'].append(dice)
                    metrics[t]['iou'].append(iou)
                    metrics[t]['precision'].append(p)
                    metrics[t]['recall'].append(r)
                    metrics[t]['acc'].append(acc)
                    metrics[t]['f1'].append(f1)
                    # pred_masks图像保存
                    # pred = (pred * 255).byte()
                    save_image(pred, os.path.join(threshold_dir, save_name))

            # 结果输出与保存
            print(f"PSNR: {torch.mean(torch.tensor(all_psnr)):.2f} ± {torch.std(torch.tensor(all_psnr)):.2f}")
            print(f"SSIM: {torch.mean(torch.tensor(all_ssim)):.4f} ± {torch.std(torch.tensor(all_ssim)):.4f}")

            # 绘制指标曲线
            plt.figure(figsize=(10, 6))
            for metric in ['dice', 'iou', 'f1', 'precision', 'recall', 'acc']:
                metric_values = [torch.mean(torch.tensor(metrics[t][metric])).item() for t in thresholds]
                plt.plot(thresholds, metric_values, label=metric)
            plt.legend()
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.savefig(f"{config.eval_save}/metrics_plot.png", dpi=300, bbox_inches='tight')

            # 计算各指标的平均值和标准差
            threshold_metrics_summary = {}
            for t in thresholds:
                threshold_metrics_summary[t] = {
                    'dice': {
                        'mean': torch.mean(torch.tensor(metrics[t]['dice'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['dice'])).item()
                    },
                    'iou': {
                        'mean': torch.mean(torch.tensor(metrics[t]['iou'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['iou'])).item()
                    },
                    'precision': {
                        'mean': torch.mean(torch.tensor(metrics[t]['precision'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['precision'])).item()
                    },
                    'recall': {
                        'mean': torch.mean(torch.tensor(metrics[t]['recall'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['recall'])).item()
                    },
                    'acc': {
                        'mean': torch.mean(torch.tensor(metrics[t]['acc'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['acc'])).item()
                    },
                    'f1': {
                        'mean': torch.mean(torch.tensor(metrics[t]['f1'])).item(),
                        'std': torch.std(torch.tensor(metrics[t]['f1'])).item()
                    }
                }

            # 保存结果到文本文件
            results = {
                'psnr_mean': torch.mean(torch.tensor(all_psnr)).item(),
                'psnr_std': torch.std(torch.tensor(all_psnr)).item(),
                'ssim_mean': torch.mean(torch.tensor(all_ssim)).item(),
                'ssim_std': torch.std(torch.tensor(all_ssim)).item(),
                'threshold_metrics': threshold_metrics_summary
            }

            save_results_to_txt(results, f"{config.eval_save}/results.txt")
            break

if __name__ == "__main__":
    evaluate()
