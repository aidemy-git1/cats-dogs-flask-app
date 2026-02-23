import os
import shutil
import random
from tqdm import tqdm

original_dataset_dir = 'PetImages'
base_dir = 'data'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

for split in ['train', 'valid']:
    for category in ['Cat', 'Dog']:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

split_ratio = 0.8

for category in ['Cat', 'Dog']:
    source_dir = os.path.join(original_dataset_dir, category)
    images = os.listdir(source_dir)
    
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)
    
    train_images = images[:split_point]
    valid_images = images[split_point:]
    
    for img in tqdm(train_images):
        src = os.path.join(source_dir, img)
        dst = os.path.join(train_dir, category, img)
        try:
            shutil.copyfile(src, dst)
        except:
            pass
    
    for img in tqdm(valid_images):
        src = os.path.join(source_dir, img)
        dst = os.path.join(valid_dir, category, img)
        try:
            shutil.copyfile(src, dst)
        except:
            pass

print("データ分割完了！")
