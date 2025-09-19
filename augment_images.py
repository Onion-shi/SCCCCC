import os
import argparse
import pandas as pd
from PIL import Image
import torch
import numpy as np
from aug.img_aug import strong_view   # 你的图像增强函数

# 工具函数：把 tensor 转成 PIL
def tensor_to_pil(x):
    """
    x: Tensor, shape (C,H,W), range 0-1
    return: PIL.Image
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().clamp(0,1)
        if x.dim() == 3:
            x = x.permute(1,2,0)  # (H,W,C)
        x = (x.numpy() * 255).astype(np.uint8)
        return Image.fromarray(x)
    else:
        raise TypeError(f"Expected torch.Tensor but got {type(x)}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    new_records = []
    for i, row in df.iterrows():
        img_path = row["image_name"] if "image_name" in row else row["image_path"]
        txt = row["description"] if "description" in row else row.get("text", "No text provided.")

        # 读取原图
        img_pil = Image.open(img_path).convert("RGB")

        # 生成 N 个增强视图
        for v in range(args.num_views):
            img_aug_tensor = strong_view(img_pil)  # 这里返回 Tensor
            img_aug_pil = tensor_to_pil(img_aug_tensor)

            # 保存增强图像
            base = os.path.splitext(os.path.basename(img_path))[0]
            new_name = f"{base}_aug{v}.png"
            new_path = os.path.join(args.output_dir, new_name)
            img_aug_pil.save(new_path)

            # 写入新 CSV 记录
            new_records.append({
                "image_path": new_path,
                "text": txt
            })

    # 保存新的 CSV
    out_csv = os.path.join(args.output_dir, "augmented.csv")
    pd.DataFrame(new_records).to_csv(out_csv, index=False)
    print(f"增强完成，结果保存到 {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\train_unlabeled_5.csv'
                        ,required=False, help="原始 CSV 文件 (包含 image_path, text)")
    parser.add_argument("--output_dir", type=str,
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_images',
                        required=False, help="增强图像输出目录")
    parser.add_argument("--num_views", type=int, default=3, help="每张图生成多少个增强视图")
    args = parser.parse_args()
    main(args)
