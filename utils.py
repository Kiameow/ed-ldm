import os
import re
def get_latest_ckpt_path(save_dir):
    if not os.path.exists(save_dir):
        return None, None
    
    # 假设所有的ckpt文件夹都在当前目录下
    ckpt_paths = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    if len(ckpt_paths) == 0:
        return None, None
    
    # 提取步骤数并找到最新的文件夹
    latest_folder = max(ckpt_paths, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # 提取epoch值
    epoch = int(re.search(r'\d+', latest_folder).group())
    
    # 返回完整的路径和epoch值
    return os.path.join(save_dir, latest_folder), epoch
