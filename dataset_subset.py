import os, shutil, random

original_dir = 'OCT2017/train'
base_dir = 'OCT_small'

# Create small dataset directory
os.makedirs(base_dir, exist_ok=True)
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    for cls in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Copy 1000 train + 500 val images per class
for cls in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
    src = os.path.join(original_dir, cls)
    all_imgs = os.listdir(src)
    random.shuffle(all_imgs)

    train_imgs = all_imgs[:1000]
    val_imgs = all_imgs[1000:1500]

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(base_dir, 'train', cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(base_dir, 'val', cls, img))
