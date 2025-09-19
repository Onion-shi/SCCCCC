# 🧪 Textmatch: Semi-supervised Medical Image Segmentation with Text Prompts

This is a PyTorch reproduction of **Textmatch** for semi-supervised medical image segmentation, based on the Mean Teacher framework.  
The implementation supports **multi-view consistency (MCR)**, **pseudo-label guided contrastive learning (PGCL)**, and a **Bilateral Prompt Decoder (BPD)** with text-image fusion.

---

## 📦 Environment

conda create -n textmatch python=3.9 -y  
conda activate textmatch  

pip install torch torchvision torchaudio  
pip install timm==0.9.2  
pip install transformers==4.33.2  
pip install scikit-learn scikit-image  
pip install opencv-python pillow matplotlib tqdm pandas  

---

## 📂 Data Preparation

Prepare datasets in CSV format:

- **Labeled data (`train_labeled_x.csv`)**

image_path,mask_path,text  
path/to/img1.png,path/to/mask1.png,Patient shows mild opacity  
path/to/img2.png,path/to/mask2.png,No abnormal findings  

- **Unlabeled data (`train_unlabeled_x.csv`)**

image_name,text  
img100.png,Opacity in right lower lobe  
img101.png,No text provided  

- **Augmented pool (`aug_pool.csv`)**

image_path,text,src_key  
path/to/aug_img100_1.png,Opacity in right lower lobe,img100.png  
path/to/aug_img100_2.png,Opacity in right lower lobe,img100.png  

- **Validation/Test (`new_test_text.csv`)**

image_path,mask_path,text  
path/to/img200.png,path/to/mask200.png,Patchy opacity in bilateral lungs  

⚠️ The `text` column **must not be empty**, otherwise the model degenerates into unimodal segmentation.

---

## 🚀 Training

Run:

python train.py \  
  --labeled_csv data/train_labeled_5.csv \  
  --unlabeled_csv data/train_unlabeled_5.csv \  
  --aug_pool_csv data/aug_pool.csv \  
  --img_root data/img/ \  
  --device cuda \  
  --epochs 400 \  
  --batch_size 8  

- Training logs (sup, reg, pl, cl, total) are saved in `train_logs.csv`.  
- Model checkpoints are saved in `ckpts/textmatch_student.pth` and `ckpts/textmatch_teacher.pth`.  
- Every `N` epochs, T-SNE visualizations of PGCL embeddings are saved in `tsne_plots/`.

---

## 🧪 Testing

Enhanced evaluation script:

python test_plus.py \  
  --val_csv data/new_test_text.csv \  
  --ckpt_teacher ckpts/textmatch_teacher.pth \  
  --use_teacher \  
  --tta \  
  --thr_sweep  

- `--use_teacher` : evaluate EMA Teacher (recommended)  
- `--tta` : enable test-time augmentation (horizontal flip)  
- `--thr_sweep` : sweep thresholds on validation set to choose the best  

Output example:

[Val] Dice=0.8231 IoU=0.7012 (N=2113)

---

## 🛠️ Modules

- `datasets.py / datasets_aug.py` : Dataset loaders & augmentations  
- `augment_images.py / augment_texts.py` : Generate image/text augmentations  
- `enc_image.py / enc_text.py` : Image encoder & text encoder (Bio_ClinicalBERT)  
- `bpd.py` : **Bilateral Prompt Decoder (BPD)** with learnable α residual  
- `pgcl.py` : **Pseudo-label Guided Contrastive Learning (PGCL)**, Eq.(10)-(11)  
- `seg_losses.py` : Dice & BCE segmentation losses  
- `train.py` : Training loop (sup, reg, pl, cl)  
- `test_plus.py` : Evaluation with Teacher/TTA/threshold sweep  

---

## 📊 Logs & Visualization

- `train_logs.csv` : Per-epoch loss values  
- Plot loss curves:

import pandas as pd, matplotlib.pyplot as plt  
df = pd.read_csv("train_logs.csv")  
plt.plot(df["epoch"], df["sup"], label="sup")  
plt.plot(df["epoch"], df["reg"], label="reg")  
plt.plot(df["epoch"], df["pl"], label="pl")  
plt.plot(df["epoch"], df["cl"], label="cl")  
plt.legend(); plt.show()  

- `tsne_plots/tsne_epX.png` : T-SNE plots of pixel embeddings, showing PGCL convergence.

---

## ❓ FAQ

- **Why is reg loss so small (~0.08)?**  
  Because teacher is EMA of student; their predictions are naturally close. A small MSE is expected.

- **Why does student test worse than teacher?**  
  In Mean Teacher frameworks, Teacher is used for evaluation because it generalizes better.

- **Why is CL loss ~0.693 at start?**  
  That’s ln(2), expected for balanced random initialization in InfoNCE.

- **Why is my CL loss always 0?**  
  If you used high-confidence mask and teacher was uncertain, all pixels got filtered. Use soft pseudo labels (default implementation).

- **Batch size too small?**  
  Use gradient accumulation to simulate larger batches, or increase epochs to ~400.

---

## 📌 Citation

If you use this repo, please cite:

@article{li2024textmatch,  
  title={Textmatch: Using Text Prompts to Improve Semi-supervised Medical Image Segmentation},  
  author={Li, ...},  
  journal={Medical Image Analysis},  
  year={2024}  
}
