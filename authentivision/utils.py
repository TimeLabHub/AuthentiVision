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
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from huggingface_hub import hf_hub_download
from .model import AdvancedFaceDetectionModel

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def transform_image(image):
    """Transform image for model input"""
    return transform(image)


def extract_glcm_features(image):
    """Extract GLCM features from image"""
    image_uint8 = (image * 255).astype(np.uint8)
    if len(image_uint8.shape) == 3:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    image_uint8 = image_uint8 // 4

    glcm = graycomatrix(
        image_uint8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True
    )

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in properties:
        features.extend(graycoprops(glcm, prop).flatten())

    return np.array(features, dtype=np.float32)


def analyze_spectrum(image, target_spectrum_length=181):
    """Analyze frequency spectrum of image"""
    if len(image.shape) == 3:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32) / 255.0

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
    """Extract edge features from image"""
    if len(image.shape) == 3:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        image = (image * 255).astype(np.uint8)

    edges = cv2.Canny(image, 100, 200)
    edges_resized = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
    return edges_resized.astype(np.float32) / 255.0


def extract_lbp_features(image):
    """Extract LBP features from image"""
    if len(image.shape) == 3:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32) / 255.0

    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform'

    lbp = local_binary_pattern(image, n_points, radius, METHOD)

    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist.astype(np.float32)


def load_model(repo_id="haijian06/AuthentiVision", filename="best_model.pth"):
    """Load model from HuggingFace Hub"""
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = AdvancedFaceDetectionModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model