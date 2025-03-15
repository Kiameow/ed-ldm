# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config
import os

class MultiConditionDataset(Dataset):
    """加载图像-文本配对数据集（支持指定数据根目录）"""

    def __init__(self, root_dir: str, config, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录路径
            config (TrainConfig): 训练配置对象
            transform (callable, optional): 图像预处理变换
        """
        super().__init__()
        self.root_dir = root_dir
        self.config = config
        self.transform = transform

        # 初始化数据路径列表
        self.img_dir = os.path.join(root_dir, "images")
        self.cond_img_dir = os.path.join(root_dir, "cond_images")
        self.text_path = os.path.join(root_dir, "texts.txt")

        # 验证目录存在性
        self._validate_paths()

        # 加载数据路径和文本 需要修改
        self.img_paths, self.cond_img_paths, self.texts= self._load_data_paths()

    def _validate_paths(self):
        """验证必要目录和文件是否存在"""
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        if not os.path.exists(self.cond_img_dir):
            raise FileNotFoundError(f"条件图像目录不存在: {self.cond_img_dir}")
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"文本文件不存在: {self.text_path}")

    def _load_data_paths(self):
        """加载数据路径和对应文本描述"""
        # 读取文本描述文件（假设格式：filename,description）
        with open(self.text_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(",", 1) for line in f.readlines()]

        img_paths = []
        cond_img_paths = []
        texts = []

        for filename, text in lines:
            img_path = os.path.join(self.img_dir, filename)
            cond_img_path = os.path.join(self.cond_img_dir, filename)

            if os.path.exists(img_path) and os.path.exists(cond_img_path):
                img_paths.append(img_path)
                cond_img_paths.append(cond_img_path)
                texts.append(text)
            else:
                print(f"警告：缺失文件 {filename}，已跳过")

        return img_paths, cond_img_paths, texts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.img_paths[idx]).convert("L")
        cond_img = Image.open(self.cond_img_paths[idx]).convert("L")

        # 应用预处理（需适配单通道）
        if self.transform:
            img = self.transform(img)
            cond_img = self.transform(cond_img)

        # 添加通道维度（如果转换未自动处理）
        if img.dim() == 2:  # [H, W]
            img = img.unsqueeze(0)  # [1, H, W]
            cond_img = cond_img.unsqueeze(0)

        # 获取文本
        text = self.texts[idx]

        return {
            "image": img,  # [C, H, W]
            "cond_image": cond_img,  # [C, H, W]
            "text": text  # str
        }


def get_dataloaders(config):
    # 初始化数据集
    train_dataset = MultiConditionDataset(
        root_dir=config.train_data_dir,
        config=config,
        transform=config.transform
    )

    test_dataset = MultiConditionDataset(
        root_dir=config.test_data_dir,
        config=config,
        transform=config.transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True  # 保持worker进程活跃
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def load_dataset(dataset_root, filter_option="both", transform=None):
    """
    Load dataset images along with their conditional images and labels, with optional image transformation.

    Parameters:
        dataset_root (str): The root directory of the dataset.
        filter_option (str): One of "healthy", "diseased", or "both".
                             - "healthy": returns only images with the 'normal' label.
                             - "diseased": returns only images with non-'normal' labels.
                             - "both": returns all images.
        transform (callable, optional): A function/transform to apply to the main image.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - "image": the loaded (and transformed, if applicable) image from the images folder.
            - "cond_image": the loaded condition image from the cond_images folder.
            - "label": the classification string.
    """
    # Define paths to the training subdirectories and text file.
    images_dir = os.path.join(dataset_root, "images")
    cond_images_dir = os.path.join(dataset_root, "cond_images")
    txt_file = os.path.join(dataset_root, "texts.txt")

    dataset = []
    
    # Read the text file and parse classification info.
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is expected to be in the format: filename,label
            parts = line.split(",")
            if len(parts) < 2:
                continue
            filename = parts[0].strip()
            label = parts[1].strip()

            # Apply filtering based on the provided option.
            if filter_option == "healthy" and label.lower() != "normal":
                continue
            if filter_option == "diseased" and label.lower() == "normal":
                continue

            # Construct full paths to the images.
            image_path = os.path.join(images_dir, filename)
            cond_image_path = os.path.join(cond_images_dir, filename)
            
            # Check if both images exist.
            if not os.path.exists(image_path) or not os.path.exists(cond_image_path):
                print(f"Warning: Missing file for {filename}. Skipping.")
                continue

            # Open the images and convert them to Grayscale.
            image = Image.open(image_path).convert("L")
            cond_image = Image.open(cond_image_path).convert("L")
            
            # Apply the transformation to the main image if provided.
            if transform is not None:
                image = transform(image)
                cond_image = transform(cond_image)
            
            # Append the loaded data to the dataset.
            dataset.append({
                "image": image,
                "cond_image": cond_image,
                "label": label
            })
    return dataset

# Example usage:
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
# dataset = load_dataset("/path/to/dataset", filter_option="both", transform=transform)
# for data in dataset:
#     print(data["label"])
