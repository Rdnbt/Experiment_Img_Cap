import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

# Initialize the Mongolian BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-base-mongolian-cased', use_fast=False)

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, delimiter='|')
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image_name"]
        self.captions = self.df["corrected_comment"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Tokenize and encode the caption
        numericalized_caption = tokenizer.encode(caption, add_special_tokens=True, max_length=512, truncation=True)

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]

        # Pad the sequences
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = tokenizer.pad_token_id

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((299, 299)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/home/eba/Dataset/exp/test_50",
        "/home/eba/Dataset/exp/50_annotations.csv",
        transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)

