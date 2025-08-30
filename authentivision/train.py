# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import multiprocessing
import random
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Import from our custom modules
from feature_extraction import AdvancedFaceDataset
from model_architecture import AdvancedFaceDetectionModel
from training_utils import train_model, evaluate, plot_metrics

# Suppress potential warning messages
warnings.filterwarnings("ignore")


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
    train_indices, temp_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
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

    # Save test set paths and labels to file
    with open('test_dataset.txt', 'w') as f:
        for i in test_indices:
            f.write(f"{dataset.image_paths[i]}\t{dataset.labels[i]}\n")
    print("Test set has been saved to 'test_dataset.txt' file.")

    # Get available CPU cores
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU cores: {cpu_count}")

    # Create DataLoaders
    batch_size = 32
    num_workers = min(cpu_count, 8)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lbp_n_bins = dataset.lbp_n_bins
    model = AdvancedFaceDetectionModel(spectrum_length=181, lbp_n_bins=lbp_n_bins).to(device)

    # TensorBoard writer
    log_dir = 'runs/advanced_face_detection'
    writer = SummaryWriter(log_dir=log_dir)

    try:
        sample_image, sample_glcm, sample_spectrum, sample_edge, sample_lbp, _ = next(iter(train_loader))
        sample_image, sample_glcm, sample_spectrum, sample_edge, sample_lbp = \
            sample_image.to(device), sample_glcm.to(device), sample_spectrum.to(device), sample_edge.to(
                device), sample_lbp.to(device)
        writer.add_graph(model, (sample_image, sample_glcm, sample_spectrum, sample_edge, sample_lbp))
    except Exception as e:
        print(f"TensorBoard graph logging failed: {e}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    num_epochs = 100
    patience = 10
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer,
                          patience=patience)

    writer.close()

    # Final evaluation on the test set
    print("\nEvaluating on the test set with the best model...")
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f"Test Set Results -> Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    else:
        print("Error: 'best_model.pth' not found. Could not evaluate on the test set.")

    # Plot metrics
    plot_metrics(history)


if __name__ == '__main__':
    main()