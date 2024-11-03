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

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        """
        Initialize the attention module.

        Args:
            in_features (int): Number of input features.
        """
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, max(in_features // 8, 1)),
            nn.ReLU(),
            nn.Linear(max(in_features // 8, 1), in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        attention_weights = self.attention(x)
        return x * attention_weights

class AdvancedFaceDetectionModel(nn.Module):
    def __init__(self, spectrum_length=181, lbp_n_bins=10):
        """
        Initialize the advanced face detection model.

        Args:
            spectrum_length (int): Fixed length for spectrum features.
            lbp_n_bins (int): Length of the LBP feature histogram.
        """
        super(AdvancedFaceDetectionModel, self).__init__()

        self.efficientnet = timm.create_model('tf_efficientnetv2_b2', pretrained=True, num_classes=0)

        self.glcm_fc = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.lbp_fc = nn.Sequential(
            nn.Linear(lbp_n_bins, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        image_feature_size = self.efficientnet.num_features
        self.image_attention = AttentionBlock(image_feature_size)
        self.glcm_attention = AttentionBlock(128)
        self.spectrum_attention = AttentionBlock(128)
        self.edge_attention = AttentionBlock(128 * 16 * 16)
        self.lbp_attention = AttentionBlock(128)

        total_features = image_feature_size + 128 + 128 + (128 * 16 * 16) + 128
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, glcm_features, spectrum_features, edge_features, lbp_features):
        """
        Forward pass.

        Args:
            image (torch.Tensor): Image tensor of shape (batch_size, 3, H, W).
            glcm_features (torch.Tensor): GLCM features of shape (batch_size, 20).
            spectrum_features (torch.Tensor): Spectrum features of shape (batch_size, spectrum_length).
            edge_features (torch.Tensor): Edge features of shape (batch_size, H, W).
            lbp_features (torch.Tensor): LBP features of shape (batch_size, lbp_n_bins).

        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
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
