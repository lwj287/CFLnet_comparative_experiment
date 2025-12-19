import os
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from dataset import ImageEnhancementDataset
from DNnet_train import DNnet
from utils.mse_utils import measure_MSEs
from utils.psnr_utils import measure_PSNRs
from utils.ssim_utils import measure_SSIMs
from utils.uiqm_utils import measure_UIQMs

def data_loader(dir_imgs1, dir_imgs2):
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    test_dataset = ImageEnhancementDataset(
        root_dir_imgs1=dir_imgs1,
        root_dir_imgs2=dir_imgs2,
        transform=test_transform,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.test_batch_size, shuffle=False)
    return test_loader

def model_loader():
    trained_model = DNnet(n=1)
    trained_model.load_state_dict(torch.load(config.trained_models_dir, weights_only=False)["model"])
    trained_model.to(config.device)
    return trained_model

def test_main(test_loader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            start_idx = batch_idx * config.test_batch_size
            end_idx = start_idx + inputs.size(0)
            current_filenames = test_loader.dataset.image_files_imgs1[start_idx:end_idx]
            for i in range(inputs.size(0)):
                output_img = outputs[i]
                raw_filename = current_filenames[i]
                save_path = os.path.join(config.enhanced_images_dir, raw_filename)
                torchvision.utils.save_image(output_img, save_path)
    MSEs = measure_MSEs(gtr_dir=config.test_dir_GT, enh_dir=config.enhanced_images_dir)
    PSNRs = measure_PSNRs(gtr_dir=config.test_dir_GT, enh_dir=config.enhanced_images_dir)
    SSIMs = measure_SSIMs(gtr_dir=config.test_dir_GT, enh_dir=config.enhanced_images_dir)
    UIQMs = measure_UIQMs(dir_name=config.enhanced_images_dir)
    config.test_mse = np.mean(MSEs)
    config.test_psnr = np.mean(PSNRs)
    config.test_ssim = np.mean(SSIMs)
    config.test_uiqm = np.mean(UIQMs)
    print(f"测试结果 >> \n MSE: {config.test_mse}\n PSNR: {config.test_psnr} dB\n SSIM: {config.test_ssim}\n UIQM: {config.test_uiqm}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model")
    parser.add_argument('--test_dir_raw', type=str, default="./a_EUVP/test_raw")
    parser.add_argument('--test_dir_GT', type=str, default="./a_EUVP/test_GT")
    parser.add_argument('--enhanced_images_dir', type=str, default="./a_compare/DNnet/enhs_EUVP")
    parser.add_argument('--trained_models_dir', type=str, default="./a_compare/DNnet/checkpoint_EUVP.pth")
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--test_mse', type=float, default=0.0)
    parser.add_argument('--test_psnr', type=float, default=0.0)
    parser.add_argument('--test_ssim', type=float, default=0.0)
    parser.add_argument('--test_uiqm', type=float, default=0.0)
    config = parser.parse_args()
    test_loader = data_loader(config.test_dir_raw, config.test_dir_GT)
    model = model_loader()
    test_main(test_loader, model)