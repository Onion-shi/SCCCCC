import os
import argparse
import pandas as pd

def main(args):
    # 读取增强图像路径的 CSV 文件
    df_img = pd.read_csv(args.aug_images_csv)

    # 读取增强文本的 CSV 文件
    text_csvs = [pd.read_csv(f) for f in args.text_csvs]
    df_txt = pd.concat(text_csvs, ignore_index=True)

    # 检查列名
    if "image" not in df_txt.columns[0].lower():
        raise ValueError("增强文本 CSV 必须包含 image_path 列")
    if "text" not in df_txt.columns[-1].lower():
        raise ValueError("增强文本 CSV 必须包含 text 列")

    # 创建一个新列表来存储最终的记录
    new_records = []
    for idx, row in df_img.iterrows():
        img_path = row["image_path"]
        # 获取原始图像的 key (去掉增强的部分)
        src_key = os.path.basename(img_path).split("_aug")[0]

        # 根据图像的增强版本匹配对应的文本
        # 根据增强版本的索引来选择对应的文本
        augmentation_index = int(os.path.basename(img_path).split("_aug")[-1][0])  # 获取增量的编号（例如 aug0 对应 0）
        matched_txt = df_txt[df_txt["image_path"].str.contains(src_key, case=False, na=False)]

        # 如果没有匹配的文本，添加默认的文本
        if matched_txt.empty:
            new_records.append({
                "image_path": img_path,
                "text": "No text provided.",
                "src_key": src_key
            })
        else:
            # 选择匹配文本中的第一个条目
            selected_txt = matched_txt.iloc[augmentation_index]["text"]
            new_records.append({
                "image_path": img_path,
                "text": selected_txt,
                "src_key": src_key
            })

    # 保存结果到输出 CSV
    out_csv = os.path.join(args.output_dir, "aug_pool.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(new_records).to_csv(out_csv, index=False)
    print(f"[OK] 已保存增强池 {out_csv}, 共 {len(new_records)} 条样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_images_csv", type=str,
                        default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_images\augmented.csv',
                        required=False,
                        help="augment_images.py 生成的 CSV (增强图像路径)")
    parser.add_argument("--text_csvs", type=str, nargs="+",
                        default=[r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_texts\aug_text_1.csv',
                                 r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_texts\aug_text_2.csv',
                                 r'D:\2025\scc\Textmatch\data\covid19\Train_Folder\aug_texts\aug_text_3.csv'
                                 ],
                        required=False,
                        help="增强后的文本 CSV 文件 (一个或多个)")
    parser.add_argument("--output_dir", type=str, default=r'D:\2025\scc\Textmatch\data\covid19\Train_Folder',required=False,
                        help="输出目录")
    args = parser.parse_args()
    main(args)
