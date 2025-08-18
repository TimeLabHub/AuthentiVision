import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import timm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from tqdm import tqdm
import multiprocessing
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Suppress potential warning messages
warnings.filterwarnings("ignore")


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, writer, patience=10):
    """
    Train the model with real-time progress display.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Computing device (CPU/GPU)
        writer: TensorBoard writer
        patience: Early stopping patience

    Returns:
        Training history dictionary
    """
    scaler = GradScaler()
    best_val_f1 = 0.0
    epochs_no_improve = 0

    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        # Use tqdm for progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for batch_idx, (image, glcm, spectrum, edge, lbp, labels) in enumerate(loop):
            # Move data to device
            image = image.to(device, non_blocking=True)
            glcm = glcm.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            lbp = lbp.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with autocast():
                outputs = model(image, glcm, spectrum, edge, lbp)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * image.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
        epoch_f1 = f1_score(all_labels, all_preds)

        # Store metrics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['train_f1'].append(epoch_f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_acc:.4f}, F1 Score: {epoch_f1:.4f}")

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('F1_Score/train', epoch_f1, epoch)

        # Validation
        val_acc, val_f1 = evaluate(model, val_loader, device, writer, epoch, phase='val')
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with validation F1 score: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation F1 score for {epochs_no_improve} consecutive epochs.")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

        # Update learning rate scheduler
        scheduler.step(val_f1)

    print("Training completed!")
    return history


def evaluate(model, data_loader, device, writer=None, epoch=0, phase='test'):
    """
    Evaluate model on given dataset with progress display.

    Args:
        model: Neural network model
        data_loader: Data loader
        device: Computing device
        writer: TensorBoard writer (optional)
        epoch: Current epoch number
        phase: Evaluation phase ('val' or 'test')

    Returns:
        Tuple of (accuracy, f1_score)
    """
    model.eval()
    all_labels = []
    all_preds = []

    # Use tqdm for progress bar
    loop = tqdm(data_loader, desc=f"Evaluating ({phase})", leave=False)

    with torch.no_grad():
        for image, glcm, spectrum, edge, lbp, labels in loop:
            # Move data to device
            image = image.to(device, non_blocking=True)
            glcm = glcm.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            lbp = lbp.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(1)

            outputs = model(image, glcm, spectrum, edge, lbp)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    # Log to TensorBoard if writer is provided
    if writer:
        writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)
        writer.add_scalar(f'F1_Score/{phase}', f1, epoch)
        writer.add_scalar(f'Precision/{phase}', precision, epoch)
        writer.add_scalar(f'Recall/{phase}', recall, epoch)

    return accuracy, f1


def plot_metrics(history):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        history: Dictionary containing training history
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy and F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'g-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(epochs, history['train_f1'], 'g--', label='Training F1 Score')
    plt.plot(epochs, history['val_f1'], 'r--', label='Validation F1 Score')
    plt.title('Accuracy and F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150)  # Save figure with higher DPI
    plt.show()


def main():
    """Main execution function."""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Enable cuDNN acceleration
    torch.backends.cudnn.benchmark = True

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define directories for real and fake face images
    real_dirs = [
        '/root/autodl-tmp/real_images/ffhq_256_6000',
        '/root/autodl-tmp/real_images/celebA_256_7200',
        '/root/autodl-tmp/real_images/Subjective Data Collection_256_2800',
        '/root/autodl-tmp/real_images/First Collection_256_5000',
        '/root/autodl-tmp/real_images/huggingface_256_9100',
        '/root/autodl-tmp/real_images/pexels_forceresize_256_1300',
        '/root/autodl-tmp/real_images/pexels_resize_256_3200',
        '/root/autodl-tmp/real_images/pixabay_forceresize_256_400'
    ]

    fake_dirs = [
        '/root/autodl-tmp/ai_generated/StyleGAN2_256_1500',
        '/root/autodl-tmp/ai_generated/ProGAN_256_2000',
        '/root/autodl-tmp/ai_generated/StyleGAN1_256_1500',
        '/root/autodl-tmp/ai_generated/Stable_256_2000',
        '/root/autodl-tmp/ai_generated/StyleGAN3_256_2000',
        '/root/autodl-tmp/ai_generated/EG3D_256_2000',
        '/root/autodl-tmp/ai_generated/Flux1_256_5000',
        '/root/autodl-tmp/ai_generated/StableDiffusion3_256_2000',
        '/root/autodl-tmp/ai_generated/DALLE2_256_1500',
        '/root/autodl-tmp/ai_generated/StableDiffusion2_256_1500',
        '/root/autodl-tmp/ai_generated/midjourneyv6_bodyface_256_13000',
        '/root/autodl-tmp/ai_generated/midjourneyv6_face_256_1000'
    ]

    # Initialize dataset
    try:
        dataset = AdvancedFaceDataset(real_dirs, fake_dirs, transform=transform)
    except ValueError as e:
        print(e)
        exit(1)

    # Get labels
    labels = dataset.labels

    # Split dataset into train (80%), validation (10%), and test (10%)
    # First split into 80% train and 20% temp (for further splitting into val and test)
    train_indices, temp_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,  # 20% for validation and test
        stratify=labels,
        random_state=42
    )

    # Split temp into validation (10%) and test (10%)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # 50% of 20% = 10%
        stratify=[labels[i] for i in temp_indices],
        random_state=42
    )

    # Create subsets
    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)
    test_data = torch.utils.data.Subset(dataset, test_indices)

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Save test set separately
    test_image_paths = [dataset.image_paths[i] for i in test_indices]
    test_labels = [dataset.labels[i] for i in test_indices]

    # Save test set paths and labels to file
    with open('test_dataset.txt', 'w') as f:
        for path, label in zip(test_image_paths, test_labels):
            f.write(f"{path}\t{label}\n")

    print("Test set has been saved to 'test_dataset.txt' file.")

    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU cores: {cpu_count}")

    # Create DataLoaders
    batch_size = 32  # Adjust as needed
    num_workers = min(cpu_count, 8)  # Adjust based on environment

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get lbp_n_bins from dataset
    lbp_n_bins = dataset.lbp_n_bins

    model = AdvancedFaceDetectionModel(spectrum_length=181, lbp_n_bins=lbp_n_bins).to(device)

    # TensorBoard writer
    log_dir = 'runs/advanced_face_detection'
    writer = SummaryWriter(log_dir=log_dir)


    try:
        sample_image, sample_glcm, sample_spectrum, sample_edge, sample_lbp, _ = next(iter(train_loader))
        sample_image = sample_image.to(device)
        sample_glcm = sample_glcm.to(device)
        sample_spectrum = sample_spectrum.to(device)
        sample_edge = sample_edge.to(device)
        sample_lbp = sample_lbp.to(device)
        writer.add_graph(model, (sample_image, sample_glcm, sample_spectrum, sample_edge, sample_lbp))
    except Exception as e:
        print(f"TensorBoard: {e}")


    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # train
    num_epochs = 100
    patience = 10
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer,
                          patience=patience)

    # close TensorBoard writer
    writer.close()


    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f"{test_acc:.4f}, F1 Score: {test_f1:.4f}")
    else:
        print("error")

    plot_metrics(history)