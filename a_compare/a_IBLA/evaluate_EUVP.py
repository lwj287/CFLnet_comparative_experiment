import os
import argparse
import cv2
import numpy as np
from utils.mse_utils import measure_MSEs
from utils.psnr_utils import measure_PSNRs
from utils.ssim_utils import measure_SSIMs
from utils.uiqm_utils import measure_UIQMs

def evaluate_quantitative_main():
    print("开始定量评估")
    print("评估结果：")
    MSEs = measure_MSEs(gtr_dir=config.gt_dir, enh_dir=config.enhanced_dir)
    print(f"MSE >> Mean: {np.mean(MSEs):.0f}, Std: {np.std(MSEs):.0f}")
    PSNRs = measure_PSNRs(gtr_dir=config.gt_dir, enh_dir=config.enhanced_dir)
    print(f"PSNR >> Mean: {np.mean(PSNRs):.3f}, Std: {np.std(PSNRs):.3f}")
    SSIMs = measure_SSIMs(gtr_dir=config.gt_dir, enh_dir=config.enhanced_dir)
    print(f"SSIM >> Mean: {np.mean(SSIMs):.3f}, Std: {np.std(SSIMs):.3f}")
    UIQMs = measure_UIQMs(dir_name=config.enhanced_dir)
    print(f"UIQM >> Mean: {np.mean(UIQMs):.3f}, Std: {np.std(UIQMs):.3f}")
    print("定量评估完毕\n")

def evaluate_qualitative_main():
    image_files = sorted([f for f in os.listdir(config.raw_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for filename in image_files:
        original_path = os.path.join(config.raw_dir, filename)
        enhanced_path = os.path.join(config.enhanced_dir, filename)
        gt_path = os.path.join(config.gt_dir, filename)
        if not os.path.exists(enhanced_path):
            print(f"警告: 增强图像 {enhanced_path} 不存在，跳过...")
            continue
        original_img = cv2.imread(original_path)
        enhanced_img = cv2.imread(enhanced_path)
        gt_img = cv2.imread(gt_path)
        original_img = cv2.resize(original_img, (256, 256))
        enhanced_img = cv2.resize(enhanced_img, (256, 256))
        gt_img = cv2.resize(gt_img, (256, 256))
        comparison_img = np.hstack((original_img, enhanced_img, gt_img))
        height, width = comparison_img.shape[:2]
        title_height = 50
        combined_img = np.zeros((height + title_height, width, 3), dtype=np.uint8)
        combined_img[title_height:, :] = comparison_img
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        img_width = width // 3
        cv2.putText(combined_img, "Original", (img_width * 0 + 10, 30), 
                    font, font_scale, color, thickness)
        cv2.putText(combined_img, "Enhanced", (img_width * 1 + 10, 30), 
                    font, font_scale, color, thickness)
        cv2.putText(combined_img, "Ground Truth", (img_width * 2 + 10, 30), 
                    font, font_scale, color, thickness)
        output_path = os.path.join(config.output_dir, f"compared_{filename}")
        cv2.imwrite(output_path, combined_img)
    print(f"所有对比图生成完成，共处理 {len(image_files)} 张图像\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model")  
    parser.add_argument('--raw_dir', type=str, default="./a_EUVP/test_raw")
    parser.add_argument('--gt_dir', type=str, default="./a_EUVP/test_GT")
    parser.add_argument('--enhanced_dir', type=str, default="./a_compare/a_IBLA/enhs_EUVP")
    parser.add_argument('--output_dir', type=str, default="./a_compare/a_IBLA/vals_EUVP")
    config = parser.parse_args()
    evaluate_quantitative_main()
    # evaluate_qualitative_main()