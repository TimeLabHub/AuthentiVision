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
# predict.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import argparse
import warnings
from tqdm import tqdm

# Import the model architecture from your file
from model_architecture import AdvancedFaceDetectionModel

# Suppress potential warning messages
warnings.filterwarnings("ignore")

# --- Feature Extraction Functions (Copied from AdvancedFaceDataset) ---
# We include them here to make the prediction script self-contained and not dependent on the dataset class.

def extract_glcm_features(image):
    """Extracts GLCM features from a grayscale numpy image."""
    image_uint8 = (image * 255).astype(np.uint8)
    image_uint8 = image_uint8 // 4
    glcm = graycomatrix(image_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    return features.astype(np.float32)

def analyze_spectrum(image, target_spectrum_length=181):
    """Analyzes frequency spectrum and extracts radial average features."""
    f = fft2(image)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    center = np.array(magnitude_spectrum.shape) // 2
    y, x = np.indices(magnitude_spectrum.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
    radial_mean = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
    if len(radial_mean) < target_spectrum_length:
        radial_mean = np.pad(radial_mean, (0, target_spectrum_length - len(radial_mean)), 'constant')
    else:
        radial_mean = radial_mean[:target_spectrum_length]
    return radial_mean.astype(np.float32)

def extract_edge_features(image):
    """Extracts edge features using Canny edge detection."""
    image_uint8 = (image * 255).astype(np.uint8)
    edges = cv2.Canny(image_uint8, 100, 200)
    edges_resized = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
    return (edges_resized.astype(np.float32) / 255.0)

def extract_lbp_features(image):
    """Extracts Local Binary Pattern (LBP) histogram features."""
    radius = 1
    n_points = 8 * radius
    n_bins = n_points + 2
    lbp = local_binary_pattern(image, n_points, radius, 'uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def get_features_for_image(image_path, transform):
    """Loads an image, applies transforms, and extracts all required features."""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Apply transforms for the model input
    tensor_image = transform(image)

    # Prepare grayscale numpy image for feature extraction
    np_image = tensor_image.cpu().numpy().transpose(1, 2, 0)
    np_image = np.clip(np_image, 0, 1) # Normalization can push values slightly outside [0,1]
    gray_image = cv2.cvtColor((np_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_image = gray_image.astype(np.float32) / 255.0

    # Extract all features
    glcm = torch.from_numpy(extract_glcm_features(gray_image))
    spectrum = torch.from_numpy(analyze_spectrum(gray_image))
    edge = torch.from_numpy(extract_edge_features(gray_image))
    lbp = torch.from_numpy(extract_lbp_features(gray_image))

    return tensor_image, glcm, spectrum, edge, lbp

def predict(model, device, image_paths, transform, threshold=0.5):
    """Runs prediction on a list of image paths."""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Predicting"):
            features = get_features_for_image(image_path, transform)
            if features is None:
                results[image_path] = "Error loading image"
                continue

            # Unpack features and add batch dimension
            img_tensor, glcm, spectrum, edge, lbp = features
            img_tensor = img_tensor.unsqueeze(0).to(device)
            glcm = glcm.unsqueeze(0).to(device)
            spectrum = spectrum.unsqueeze(0).to(device)
            edge = edge.unsqueeze(0).to(device)
            lbp = lbp.unsqueeze(0).to(device)

            # Get model output
            output = model(img_tensor, glcm, spectrum, edge, lbp)
            prob = torch.sigmoid(output).item()
            prediction = "Fake" if prob > threshold else "Real"
            
            results[image_path] = (prediction, prob)
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Advanced Face Authenticity Detection")
    parser.add_argument('--input-path', type=str, required=True, help='Path to an image file or a directory of images.')
    parser.add_argument('--model-path', type=str, default='best_model.pth', help='Path to the trained model weights.')
    parser.add_argument('--device', type=str, default='auto', help="Device to use: 'cpu', 'cuda', or 'auto' for automatic detection.")

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    # Load model
    # Note: lbp_n_bins is hardcoded based on the dataset class (radius=1 -> n_points=8 -> n_bins=10)
    model = AdvancedFaceDetectionModel(lbp_n_bins=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully.")

    # Define the same transforms as validation/test (no augmentations)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of image paths
    if os.path.isdir(args.input_path):
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    elif os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory.")
        return
        
    if not image_paths:
        print("No images found at the specified path.")
        return

    # Run prediction
    predictions = predict(model, device, image_paths, transform)

    # Print results
    print("\n--- Prediction Results ---")
    for path, result in predictions.items():
        if isinstance(result, tuple):
            prediction, prob = result
            label = f"{prediction} (Confidence: {prob:.4f})"
        else:
            label = result # Error message
        print(f"{os.path.basename(path):<40} -> {label}")


if __name__ == '__main__':
    main()
