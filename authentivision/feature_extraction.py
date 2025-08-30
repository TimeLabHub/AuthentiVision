# feature_extraction.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2

class AdvancedFaceDataset(Dataset):
    """
    Custom dataset for advanced face detection with multiple feature extraction methods.
    """

    def __init__(self, real_dirs, fake_dirs, transform=None, target_spectrum_length=181):
        """
        Initialize dataset by collecting image paths and labels.

        Args:
            real_dirs: List of directories containing real face images
            fake_dirs: List of directories containing fake face images
            transform: Torchvision transforms to apply to images
            target_spectrum_length: Target length for spectrum features
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.target_spectrum_length = target_spectrum_length

        # LBP (Local Binary Pattern) parameters
        self.lbp_radius = 1
        self.lbp_n_points = 8 * self.lbp_radius
        self.lbp_n_bins = self.lbp_n_points + 2  # For 'uniform' mode

        # Collect real images
        for directory in real_dirs:
            if not os.path.isdir(directory):
                print(f"Warning: Directory {directory} does not exist.")
                continue
            real_images = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if self._is_image_file(f)
            ]
            self.image_paths.extend(real_images)
            self.labels.extend([0] * len(real_images))  # Label 0 for real

        # Collect fake images
        for directory in fake_dirs:
            if not os.path.isdir(directory):
                print(f"Warning: Directory {directory} does not exist.")
                continue
            fake_images = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if self._is_image_file(f)
            ]
            self.image_paths.extend(fake_images)
            self.labels.extend([1] * len(fake_images))  # Label 1 for fake

        if len(self.image_paths) == 0:
            raise ValueError("No images found. Please check the provided directories.")

    def _is_image_file(self, filename):
        """
        Check if a file is an image based on its extension.

        Args:
            filename: Name of the file to check

        Returns:
            Boolean indicating if file is an image
        """
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def extract_glcm_features(self, image):
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) features from grayscale image.

        Args:
            image: Grayscale image as numpy array

        Returns:
            GLCM feature vector
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        # Quantize to 64 levels
        image_uint8 = image_uint8 // 4  # Range: [0, 63]

        # Compute GLCM
        glcm = graycomatrix(
            image_uint8,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=64,
            symmetric=True,
            normed=True
        )

        # Compute GLCM properties
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()

        # Concatenate all features
        features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
        return features.astype(np.float32)

    def analyze_spectrum(self, image):
        """
        Analyze frequency spectrum of grayscale image and extract radial average features.

        Args:
            image: Grayscale image as numpy array

        Returns:
            Radial spectrum features
        """
        # Compute 2D FFT
        f = fft2(image)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)  # Avoid log(0)

        # Compute radial distances
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.indices(magnitude_spectrum.shape)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        # Compute radial average
        radial_mean = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())

        # Ensure fixed length
        if len(radial_mean) < self.target_spectrum_length:
            radial_mean = np.pad(
                radial_mean,
                (0, self.target_spectrum_length - len(radial_mean)),
                'constant'
            )
        else:
            radial_mean = radial_mean[:self.target_spectrum_length]

        return radial_mean.astype(np.float32)

    def extract_edge_features(self, image):
        """
        Extract edge features using Canny edge detection.

        Args:
            image: Grayscale image as numpy array

        Returns:
            Edge feature map
        """
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        edges = cv2.Canny(image_uint8, 100, 200)
        # Resize to fixed size to reduce dimensions
        edges_resized = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
        return edges_resized.astype(np.float32) / 255.0  # Normalize to [0,1]

    def extract_lbp_features(self, image):
        """
        Extract Local Binary Pattern (LBP) features from grayscale image.

        Args:
            image: Grayscale image as numpy array

        Returns:
            LBP histogram features
        """
        # Use class attribute LBP parameters
        radius = self.lbp_radius
        n_points = self.lbp_n_points
        METHOD = 'uniform'

        # Compute LBP image
        lbp = local_binary_pattern(image, n_points, radius, METHOD)

        # Compute LBP histogram
        n_bins = self.lbp_n_bins  # Ensure consistent bin count
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist.astype(np.float32)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, glcm_features, spectrum_features, edge_features, lbp_features, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert to NumPy array for feature extraction
        np_image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        np_image = np.clip(np_image, 0, 1)  # Ensure values are in [0,1]

        # Convert to grayscale
        gray_image = cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_image = gray_image.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Extract features
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