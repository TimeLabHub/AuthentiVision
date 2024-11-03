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
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
from .utils import (
    transform_image,
    extract_glcm_features,
    analyze_spectrum,
    extract_edge_features,
    extract_lbp_features,
)


class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, max(in_features // 8, 1)),
            nn.ReLU(),
            nn.Linear(max(in_features // 8, 1), in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class AdvancedFaceDetectionModel(nn.Module):
    def __init__(self, spectrum_length=181, lbp_n_bins=10):
        super(AdvancedFaceDetectionModel, self).__init__()

        self.efficientnet = timm.create_model('tf_efficientnetv2_b2', pretrained=True, num_classes=0)
        for param in self.efficientnet.conv_stem.parameters():
            param.requires_grad = False
        for param in self.efficientnet.bn1.parameters():
            param.requires_grad = False

        # Define model components
        self.glcm_fc = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.lbp_fc = nn.Sequential(
            nn.Linear(lbp_n_bins, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Attention blocks
        image_feature_size = self.efficientnet.num_features
        self.image_attention = AttentionBlock(image_feature_size)
        self.glcm_attention = AttentionBlock(64)
        self.spectrum_attention = AttentionBlock(64)
        self.edge_attention = AttentionBlock(32 * 8 * 8)
        self.lbp_attention = AttentionBlock(64)

        # Final fusion layers
        total_features = image_feature_size + 64 + 64 + (32 * 8 * 8) + 64
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, glcm_features, spectrum_features, edge_features, lbp_features):
        image_features = self.efficientnet(image)
        image_features = self.image_attention(image_features)

        glcm_features = self.glcm_fc(glcm_features)
        glcm_features = self.glcm_attention(glcm_features)

        spectrum_features = self.spectrum_conv(spectrum_features.unsqueeze(1))
        spectrum_features = spectrum_features.squeeze(2)
        spectrum_features = self.spectrum_attention(spectrum_features)

        edge_features = self.edge_conv(edge_features.unsqueeze(1))
        edge_features = edge_features.view(edge_features.size(0), -1)
        edge_features = self.edge_attention(edge_features)

        lbp_features = self.lbp_fc(lbp_features)
        lbp_features = self.lbp_attention(lbp_features)

        combined_features = torch.cat(
            (image_features, glcm_features, spectrum_features, edge_features, lbp_features), dim=1
        )

        output = self.fusion(combined_features)
        return output.squeeze(1)


class AuthentiVision:
    def __init__(self, model_path=None):
        """
        Initialize AuthentiVision detector

        Args:
            model_path: Optional path to model weights. If None, downloads from HuggingFace
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path is None:
            from .utils import load_model
            self.model = load_model()
        else:
            self.model = AdvancedFaceDetectionModel()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        """
        Predict if an image contains a real face or an AI-generated face

        Args:
            image_path: Path to image or PIL Image object

        Returns:
            str: "Real Face" or "AI-Generated Face"
            float: Confidence score
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path

        # Transform image
        image_tensor = transform_image(image)
        image_tensor = image_tensor.to(self.device)

        # Extract all features
        with torch.no_grad():
            # Process image for feature extraction
            np_image = image_tensor.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            np_image = np.clip(np_image, 0, 1)

            # Extract features
            glcm_features = torch.from_numpy(extract_glcm_features(np_image)).unsqueeze(0).to(self.device)
            spectrum_features = torch.from_numpy(analyze_spectrum(np_image)).unsqueeze(0).to(self.device)
            edge_features = torch.from_numpy(extract_edge_features(np_image)).unsqueeze(0).to(self.device)
            lbp_features = torch.from_numpy(extract_lbp_features(np_image)).unsqueeze(0).to(self.device)

            # Get prediction
            outputs = self.model(image_tensor.unsqueeze(0), glcm_features, spectrum_features,
                                 edge_features, lbp_features)
            score = torch.sigmoid(outputs).item()

        label = "Real Face" if score < 0.5 else "AI-Generated Face"
        return label, score