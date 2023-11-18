import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

# Load Mongolian BERT tokenizer
tokenizer_mon = AutoTokenizer.from_pretrained("tugstugi/bert-base-mongolian-cased", use_fast=False)
BERT_VOCAB_SIZE = tokenizer_mon.vocab_size


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_mon(self, text):
        # Use Mongolian BERT tokenizer
        return tokenizer_mon.tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_mon(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_mon(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, delimiter='|')
        self.transform = transform

        self.imgs = self.df["image_name"]
        self.captions = self.df["corrected_comment"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Tokenize using BERT tokenizer
        numericalized_caption = tokenizer_mon.encode_plus(
            caption,
            max_length=32,  # Adjust max_length if needed
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].squeeze(0)  # Extract input_ids and remove the extra dimension

        # Replace out-of-range tokens with the unknown token ID
        unknown_token_id = tokenizer_mon.unk_token_id
        numericalized_caption = torch.where(
            numericalized_caption < tokenizer_mon.vocab_size,
            numericalized_caption,
            torch.tensor(unknown_token_id)
        )

        return img, numericalized_caption

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
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
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

"""
if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/home/eba/Dataset/exp/test_50",
        "/home/eba/Dataset/exp/50_annotations.csv",
        transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
"""
if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )

    loader, dataset = get_loader(
        "/home/eba/Dataset/exp/test_50",
        "/home/eba/Dataset/exp/50_annotations.csv",
        transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(f"Batch {idx} - Imgs shape: {imgs.shape}, Captions shape: {captions.shape}")
        if idx == 0:  # Only print the first batch for brevity
            print(captions)
            break
