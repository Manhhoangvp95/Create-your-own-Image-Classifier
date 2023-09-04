# Import libraries
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
# Define command line arguments
from torch.utils.data import DataLoader
from torchvision.models import DenseNet121_Weights, VGG16_Weights

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

args, _ = parser.parse_known_args()

# Define the mean and standard deviation for normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
print_every = 40

use_gpu = args.gpu
data_dir = args.data_dir
learning_rate = args.learning_rate
epochs = args.epochs
arch = args.arch
hidden_units = args.hidden_units


# Function to create data transforms
def create_data_transforms(phase):
    """
        Create data transformations based on the given phase (train, valid, or test).

        Args:
            phase (str): The phase of data processing (train, valid, or test).

        Returns:
            torchvision.transforms.Compose: A composition of data transformations.
        """
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def create_data_loaders(data_dir: str, batch_size: int = 64):
    """
    Function to create data loaders.

    Args:
        data_dir (str): The directory where the data is stored.
        batch_size (int): The batch size to use for the data loaders. Default: 64.

    Returns:
        Tuple[Dict[str, DataLoader], Dict[str, ImageFolder]]: A tuple containing two dictionaries. The first dictionary
        contains the data loaders and the second dictionary contains the image folders.
    """
    data_transforms = {
        'train': create_data_transforms('train'),
        'valid': create_data_transforms('valid'),
        'test': create_data_transforms('test')
    }

    image_datasets = {
        x: datasets.ImageFolder(data_dir + '/' + x, transform=data_transforms[x])
        for x in data_transforms.keys()
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=True if x == 'train' else False)
        for x in data_transforms.keys()
    }

    return dataloaders, image_datasets


def build_model(arch: str, hidden_units: int = 4096, dropout: float = 0.2):
    """
    Function to build a model.

    Args:
        arch (str): The architecture of the model to build. Must be one of 'vgg16' or 'densenet121'.
        hidden_units (int): The number of hidden units to use in the classifier. Default: 4096.
        dropout (float): The dropout probability to use in the classifier. Default: 0.2.

    Returns:
        Tuple[nn.Module, nn.Module]: A tuple containing two modules. The first module is the model and the second module
        is the criterion.
    """
    # Dictionary to map model names to their constructors and input sizes
    model_info = {
        'vgg16': (models.vgg16, 25088, VGG16_Weights.DEFAULT),
        'densenet121': (models.densenet121, 1024, DenseNet121_Weights.DEFAULT)
    }

    if arch in model_info:
        model_constructor, input_size, weights = model_info[arch]
        model = model_constructor(weights=weights)
    else:
        raise ValueError("Unsupported architecture:", arch)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create a custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    return model, criterion


def validate(model: nn.Module, criterion, data_loader: DataLoader, device: torch.device):
    '''
    Validate a trained deep learning model on a dataset.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        data_loader (DataLoader): DataLoader for validation data.
        device (torch.device): The device to use for validation (e.g., CPU or GPU).

    Returns:
        Tuple[float, float]: A tuple containing the validation loss and accuracy.
    '''

    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    accuracy = 0

    with torch.set_grad_enabled(False):
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model.forward(inputs)
            val_loss += criterion(output, labels).item()
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])

            # Calculate accuracy
            accuracy += equality.type(torch.FloatTensor).mean().item()

    # Calculate average validation loss and accuracy
    val_loss /= len(data_loader)
    accuracy /= len(data_loader)

    return val_loss, accuracy


def train(model: nn.Module, num_epochs: int, criterion, optimizer, train_loader: DataLoader,
          valid_loader: DataLoader, device: torch.device, print_every: int = 50):
    """
    Trains a neural network model.

    Args:
        model (torch.nn.Module): The neural network model to train.
        num_epochs (int): Number of training epochs.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimization algorithm (e.g., SGD or Adam).
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        device: Whether to use GPU for training.
        print_every (int): Frequency of printing training progress.

    Returns:
        None
    """
    model.train()  # Set the model to training mode

    steps = 0
    running_loss = 0

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            steps += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss, accuracy = validate(model, criterion, valid_loader, device)

                print(f"Epoch {epoch + 1}/{num_epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {val_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")

                running_loss = 0

                # Set the model back to training mode
                model.train()


def save_checkpoint(train_data, model, path='checkpoint.pth', arch='densenet121', hidden_units=4096, dropout=0.2,
                    lr=0.001, epochs=1):
    """
    Save a checkpoint of the trained model.

    Args:
        train_data (torchvision.datasets.ImageFolder): Training data used to map class indices to class labels.
        model (torch.nn.Module): The trained model to save.
        path (str): Path where the checkpoint file will be saved.
        arch (str): Model architecture (default: 'vgg16').
        hidden_units (int): Number of hidden units in the classifier (default: 4096).
        dropout (float): Dropout probability in the classifier (default: 0.2).
        lr (float): Learning rate used during training (default: 0.001).
        epochs (int): Number of training epochs (default: 1).

    Returns:
        None
    """
    # Set the class to index mapping in the model for later inference
    model.class_to_idx = train_data.class_to_idx

    # Create a dictionary to store model and training-related information
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': lr,
        'dropout': dropout,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    # Save the checkpoint to the specified path
    torch.save(checkpoint, path)


if __name__ == "__main__":

    # Load the data
    dataloaders, image_datasets = create_data_loaders(data_dir, batch_size=64)

    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Building and training the classifier
    model, criterion = build_model(arch, hidden_units)
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    train(model, epochs, criterion, optimizer, dataloaders['train'], dataloaders['valid'], device=device)

    # Testing network
    test_loss, accuracy = validate(model, criterion, dataloaders['test'], device)
    print("Val. Accuracy: {:.3f}".format(accuracy))
    print("Val. Loss: {:.3f}".format(test_loss))
    # Save the checkpoint
    save_checkpoint(image_datasets['train'], model, arch=arch, hidden_units=hidden_units, lr=learning_rate,
                    epochs=epochs)
