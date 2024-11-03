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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    """
    Train the model and display real-time progress.

    Args:
        model (nn.Module): Neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        patience (int): Early stopping patience.

    Returns:
        dict: Training and validation loss and accuracy history.
    """
    scaler = GradScaler()
    best_val_acc = 0.0
    epochs_no_improve = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch_idx, (image, glcm, spectrum, edge, lbp, labels) in enumerate(loop):
            image = image.to(device, non_blocking=True)
            glcm = glcm.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            lbp = lbp.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).squeeze(1)

            optimizer.zero_grad()

            with autocast():
                outputs = model(image, glcm, spectrum, edge, lbp)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * image.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), accuracy=(correct / total))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        val_acc = evaluate(model, val_loader, device)
        history['val_acc'].append(val_acc)
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        scheduler.step()

    print("Training finished!")
    return history

def evaluate(model, data_loader, device):
    """
    Evaluate the model on a given dataset and display progress.

    Args:
        model (nn.Module): Neural network model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to evaluate on.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()
    correct = 0
    total = 0

    loop = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for image, glcm, spectrum, edge, lbp, labels in loop:
            image = image.to(device, non_blocking=True)
            glcm = glcm.to(device, non_blocking=True)
            spectrum = spectrum.to(device, non_blocking=True)
            edge = edge.to(device, non_blocking=True)
            lbp = lbp.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(1)

            outputs = model(image, glcm, spectrum, edge, lbp)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(accuracy=(correct / total))

    return correct / total
