import os
from PIL import Image
from torch.utils.data import Dataset
import re

def natural_sort_key(filename):
    """提取文件名中的数字部分作为排序键"""
    numbers = re.findall(r'\d+', filename)  # 提取所有数字部分
    return [int(num) for num in numbers]    # 转换为整数列表作为排序依据

class ImageEnhancementDataset(Dataset):
    def __init__(self, root_dir_imgs1, root_dir_imgs2, transform=None):
        self.root_dir_imgs1 = root_dir_imgs1
        self.root_dir_imgs2 = root_dir_imgs2
        self.transform = transform
        
        # 获取并排序图像文件
        self.image_files_imgs1 = [f for f in os.listdir(root_dir_imgs1) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files_imgs1.sort(key=natural_sort_key)
        self.image_files_imgs2 = [f for f in os.listdir(root_dir_imgs2) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files_imgs2.sort(key=natural_sort_key)

        # 确保输入和目标图像数量一致
        assert len(self.image_files_imgs1) == len(self.image_files_imgs2), "输入和目标图像数量不匹配"

    def __len__(self):
        return len(self.image_files_imgs1)

    def __getitem__(self, idx):
        img_path_imgs1 = os.path.join(self.root_dir_imgs1, self.image_files_imgs1[idx])
        image_imgs1 = Image.open(img_path_imgs1).convert('RGB')
        img_path_imgs2 = os.path.join(self.root_dir_imgs2, self.image_files_imgs2[idx])
        image_imgs2 = Image.open(img_path_imgs2).convert('RGB')

        # 应用相同的随机变换到输入和目标
        if self.transform:
            input_image_imgs1 = self.transform(image_imgs1)
            input_image_imgs2 = self.transform(image_imgs2)

        return input_image_imgs1, input_image_imgs2