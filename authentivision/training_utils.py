# training_utils.py

import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

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