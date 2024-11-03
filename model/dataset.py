# Copyright 2024 TimeLabHub. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
from PIL import Image

class AdvancedFaceDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None, target_spectrum_length=181):
        """
        Initialize the dataset by collecting image paths and labels.

        Args:
            real_dirs (list): List of directories containing real face images.
            fake_dirs (list): List of directories containing AI-generated face images.
            transform (callable, optional): Transformations to apply to images.
            target_spectrum_length (int): Fixed length for spectrum features.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.target_spectrum_length = target_spectrum_length

        self.lbp_radius = 1
        self.lbp_n_points = 8 * self.lbp_radius
        self.lbp_n_bins = self.lbp_n_points + 2

        for dir in real_dirs:
            if not os.path.isdir(dir):
                print(f"Warning: Directory {dir} does not exist.")
                continue
            real_images = [os.path.join(dir, f) for f in os.listdir(dir) if self._is_image_file(f)]
            self.image_paths.extend(real_images)
            self.labels.extend([0] * len(real_images))

        for dir in fake_dirs:
            if not os.path.isdir(dir):
                print(f"Warning: Directory {dir} does not exist.")
                continue
            fake_images = [os.path.join(dir, f) for f in os.listdir(dir) if self._is_image_file(f)]
            self.image_paths.extend(fake_images)
            self.labels.extend([1] * len(fake_images))

        if len(self.image_paths) == 0:
            raise ValueError("No images found. Please check the provided directories.")

    def _is_image_file(self, filename):
        """Check if a file is an image based on its extension."""
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_paths)

    def extract_glcm_features(self, image):
        """
        Extract Gray-Level Co-occurrence Matrix (GLCM) features from a grayscale image.

        Args:
            image (np.ndarray): Grayscale image with values in [0, 1].

        Returns:
            np.ndarray: Flattened GLCM features.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        image_uint8 = image_uint8 // 4

        glcm = graycomatrix(
            image_uint8,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=64,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()

        features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
        return features.astype(np.float32)

    def analyze_spectrum(self, image):
        """
        Analyze the frequency spectrum of a grayscale image and extract radial average features.

        Args:
            image (np.ndarray): Grayscale image with values in [0, 1].

        Returns:
            np.ndarray: Fixed-length radial average magnitude spectrum.
        """
        f = fft2(image)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.indices(magnitude_spectrum.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

        radial_mean = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())

        if len(radial_mean) < self.target_spectrum_length:
            radial_mean = np.pad(radial_mean, (0, self.target_spectrum_length - len(radial_mean)), 'constant')
        else:
            radial_mean = radial_mean[:self.target_spectrum_length]

        return radial_mean.astype(np.float32)

    def extract_edge_features(self, image):
        """
        Extract edge features from a grayscale image using Canny edge detection.

        Args:
            image (np.ndarray): Grayscale image with values in [0, 1].

        Returns:
            np.ndarray: Edge-detected image.
        """
        image_uint8 = (image * 255).astype(np.uint8)
        edges = cv2.Canny(image_uint8, 100, 200)
        return edges.astype(np.float32) / 255.0

    def extract_lbp_features(self, image):
        """
        Extract Local Binary Pattern (LBP) features from a grayscale image.

        Args:
            image (np.ndarray): Grayscale image with values in [0, 1].

        Returns:
            np.ndarray: LBP feature histogram.
        """
        radius = self.lbp_radius
        n_points = self.lbp_n_points
        METHOD = 'uniform'

        lbp = local_binary_pattern(image, n_points, radius, METHOD)

        n_bins = self.lbp_n_bins
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist.astype(np.float32)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Transformed image tensor, GLCM features, spectrum features, edge features, LBP features, and label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        np_image = image.cpu().numpy().transpose(1, 2, 0)
        np_image = np.clip(np_image, 0, 1)

        gray_image = cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_image = gray_image.astype(np.float32) / 255.0

        glcm_features = self.extract_glcm_features(gray_image)
        spectrum_features = self.analyze_spectrum(gray_image)
        edge_features = self.extract_edge_features(gray_image)
        lbp_features = self.extract_lbp_features(gray_image)

        return (
            image,
            torch.from_numpy(glcm_features),
            torch.from_numpy(spectrum_features),
            torch.from_numpy(edge_features),
            torch.from_numpy(lbp_features),
            torch.FloatTensor([label])
        )
