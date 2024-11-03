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
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import multiprocessing
import warnings
from dataset import AdvancedFaceDataset
from model import AdvancedFaceDetectionModel
from loss import FocalLoss
from train import train_model, evaluate
from utils import plot_metrics

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dirs = [
        '/content/drive/MyDrive/data_512/real_face512',
        '/content/drive/MyDrive/face_b/0'
    ]
    fake_dirs = [
        '/content/drive/MyDrive/data_512/ai_face_512',
        '/content/drive/MyDrive/face_b/1'
    ]

    try:
        dataset = AdvancedFaceDataset(real_dirs, fake_dirs, transform=transform)
    except ValueError as e:
        print(e)
        exit(1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    cpu_count = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {cpu_count}")

    batch_size = 64
    num_workers = min(cpu_count, 16)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lbp_n_bins = dataset.lbp_n_bins

    model = AdvancedFaceDetectionModel(spectrum_length=181, lbp_n_bins=lbp_n_bins).to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    num_epochs = 50
    patience = 5
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=patience)

    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('./model/best_model.pth'))
        final_val_acc = evaluate(model, val_loader, device)
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    else:
        print("Best model not found. Please ensure training was successful.")

    plot_metrics(history)
