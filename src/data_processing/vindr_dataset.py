import os

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class VinDrDataset(Dataset):
    """
    A PyTorch Dataset class for the VinDr-Mammo dataset.
    This class handles loading and preprocessing of mammography images
    and their corresponding annotations.
    """

    def __init__(self, images_dir: str, annotations_file: str, split: str, view: str, transform=None):
        """
        Initializes the VinDrDataset.

        Args:
            images_dir (str): Directory containing mammography images.
            annotations_file (str): Path to the annotations file.
            split (str): Dataset split to use ('training', 'test').
            view (str): View type to filter images ('CC', 'MLO', etc.).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.split = split
        self.view = view
        self.transform = transform
        self.image_paths, self.annotations = self._load_data()
        
        self.to_tensor = transforms.ToTensor()
        
    def _load_data(self):
        """
        Loads image paths and annotations from the dataset.

        Returns:
            image_paths (list): List of image file paths.
            annotations (list): List of corresponding annotations.
        """
        image_paths = []
        annotations = []
        
        # Load annotations from CSV file
        df = pd.read_csv(self.annotations_file)
        for _, row in df.iterrows():
            study_id = row['study_id']
            annotation = row['breast_birads']
            split = row['split']
            view = row['view_position']
            
            image_path = os.path.join(self.images_dir, study_id)
            if os.path.exists(image_path) and split == self.split and view == self.view:
                # For each image file in the study directory, add a corresponding
                # annotation entry so image_paths and annotations remain aligned.
                files_added = 0
                for img_file in os.listdir(image_path):
                    if img_file.lower().endswith('.png'):
                        image_paths.append(os.path.join(image_path, img_file))
                        annotations.append(annotation)
                        files_added += 1
                if files_added == 0:
                    # directory exists but no pngs found; warn and continue
                    print(f"Warning: no PNG images found in {image_path}")

        return image_paths, annotations
    
    def _load_image(self, image_path):
        """
        Loads an image from the specified path.

        Args:
            image_path (str): Path to the image png.
        Returns:
            image: Tensor representation of the image.
        """
        
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        """
        Retrieves the image and annotation at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image and its annotation.
        """
        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]
        
        image = self._load_image(image_path)

        return {"image": image, "breast_birads": annotation}
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.image_paths)