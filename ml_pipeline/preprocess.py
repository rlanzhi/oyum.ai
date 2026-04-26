import os
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import Dataset

class KyrgyzOrnamentDataset(Dataset):
    def __init__(self, annotations_path, images_root, tokenizer=None, transform=None):
        self.df = pd.read_csv(annotations_path)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.transform = transform if transform else self._default_transform()

    def _default_transform(self):
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = ToTensorV2()(image=image_np)['image']

        # Build descriptive prompt
        prompt = (f"A Kyrgyz ornament, {row['motif_type']}, "
                  f"{row['symmetry_class']} symmetry, {row['color_scheme']} colors. "
                  f"{row['cultural_meaning']}")

        if self.tokenizer:
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            return {
                "pixel_values": image_tensor,
                "input_ids": tokens.input_ids.squeeze(0),
                "attention_mask": tokens.attention_mask.squeeze(0),
                "prompt": prompt
            }
        else:
            return {
                "pixel_values": image_tensor,
                "prompt": prompt
            }

if __name__ == "__main__":
    dataset = KyrgyzOrnamentDataset("../data/annotations.csv", "../data")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Image tensor shape: {sample['pixel_values'].shape}, dtype: {sample['pixel_values'].dtype}")