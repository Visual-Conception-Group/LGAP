import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file, caption_file=None, transform=None, num=None):
        self.image_folder = image_folder
        self.label_file = label_file
        self.caption_file = caption_file
        self.transform = transform
        self.num = num
        self.image_paths, self.labels, self.captions = self.load_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.caption_file:
            caption = self.captions[index]
            return image, label, caption
        else:
            return image, label

    def load_data(self):
        image_paths = []
        labels = []
        captions = []

        with open(self.label_file, 'r') as file:
            for line in file:
                label = int(line.strip())
                labels.append(label)
        
        if self.caption_file:
            with open(self.caption_file, 'r') as file:
                for line in file:
                    caption = line.strip()
                    captions.append(caption)
        num = len(labels) if self.num is None else self.num
        for i in range(num):
            image_paths.append(os.path.join(self.image_folder, "{}.png".format(i)))
        labels = labels[0:len(image_paths)]
        captions = captions[0:len(image_paths)]

        

        return image_paths, labels, captions
