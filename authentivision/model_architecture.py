# model_architecture.py

import torch
import torch.nn as nn
import timm

class AttentionBlock(nn.Module):
    """
    Attention mechanism module for feature weighting.
    """

    def __init__(self, in_features):
        """
        Initialize attention module.

        Args:
            in_features: Number of input features
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
        Forward pass through attention module.

        Args:
            x: Input tensor

        Returns:
            Weighted features
        """
        attention_weights = self.attention(x)
        return x * attention_weights


class AdvancedFaceDetectionModel(nn.Module):
    """
    Advanced face detection model with multi-modal feature fusion.
    """

    def __init__(self, spectrum_length=181, lbp_n_bins=10):
        """
        Initialize the advanced face detection model.

        Args:
            spectrum_length: Length of spectrum features
            lbp_n_bins: Number of LBP histogram bins
        """
        super(AdvancedFaceDetectionModel, self).__init__()

        # EfficientNetV2-B2 backbone
        self.efficientnet = timm.create_model('tf_efficientnetv2_b2', pretrained=True, num_classes=0)

        # Freeze initial layers
        for param in self.efficientnet.conv_stem.parameters():
            param.requires_grad = False
        for param in self.efficientnet.bn1.parameters():
            param.requires_grad = False

        # GLCM feature processing
        self.glcm_fc = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Spectrum feature processing
        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Edge feature processing
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # LBP feature processing
        self.lbp_fc = nn.Sequential(
            nn.Linear(lbp_n_bins, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Attention mechanisms
        image_feature_size = self.efficientnet.num_features  # Typically 1408
        self.image_attention = AttentionBlock(image_feature_size)
        self.glcm_attention = AttentionBlock(64)
        self.spectrum_attention = AttentionBlock(64)
        self.edge_attention = AttentionBlock(32 * 8 * 8)
        self.lbp_attention = AttentionBlock(64)

        # Fusion layers
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
        """
        Forward pass through the model.

        Args:
            image: Input image tensor
            glcm_features: GLCM feature tensor
            spectrum_features: Spectrum feature tensor
            edge_features: Edge feature tensor
            lbp_features: LBP feature tensor

        Returns:
            Model predictions
        """
        # Image features
        image_features = self.efficientnet(image)  # (batch_size, image_feature_size)
        image_features = self.image_attention(image_features)  # Apply attention

        # GLCM features
        glcm_features = self.glcm_fc(glcm_features)  # (batch_size, 64)
        glcm_features = self.glcm_attention(glcm_features)  # Apply attention

        # Spectrum features
        spectrum_features = self.spectrum_conv(spectrum_features.unsqueeze(1))  # (batch, 64, 1)
        spectrum_features = spectrum_features.squeeze(2)  # (batch, 64)
        spectrum_features = self.spectrum_attention(spectrum_features)  # Apply attention

        # Edge features
        edge_features = self.edge_conv(edge_features.unsqueeze(1))  # (batch, 32, 8, 8)
        edge_features = edge_features.view(edge_features.size(0), -1)  # (batch, 32*8*8)
        edge_features = self.edge_attention(edge_features)  # Apply attention

        # LBP features
        lbp_features = self.lbp_fc(lbp_features)  # (batch_size, 64)
        lbp_features = self.lbp_attention(lbp_features)  # Apply attention

        # Concatenate all features
        combined_features = torch.cat(
            (image_features, glcm_features, spectrum_features, edge_features, lbp_features),
            dim=1
        )

        # Fusion and output
        output = self.fusion(combined_features)  # (batch, 1)
        return output.squeeze(1)  # (batch,)