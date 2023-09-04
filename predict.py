import argparse
import json
import os.path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

# Define command line arguments
from ImageClassifier.train import build_model, MEAN, STD

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Path to the image file')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint file')
parser.add_argument('--topk', type=int, help='Top K most likely classes')
parser.add_argument('--labels', type=str, help='Path to the JSON file containing class labels')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = parser.parse_known_args()

image = args.image
use_gpu = args.gpu


def load_model(path='checkpoint.pth'):
    """
    Load a trained model from a checkpoint file.

    Args:
        path (str): Path to the checkpoint file to load (default: 'checkpoint.pth').

    Returns:
        model (torch.nn.Module): The loaded model.
        criterion: Loss function corresponding to the model.
    """
    if not os.path.exists(path):
        raise UserWarning('Checkpoint path does not exist. Please recheck the checkpoint parameter.')
    # Load the checkpoint
    checkpoint = torch.load(path)

    # Retrieve architecture and hidden_units information
    hidden_units = checkpoint['hidden_units']
    arch = checkpoint['arch']

    # Build a new model with the same architecture and hidden_units
    model, criterion = build_model(arch=arch, hidden_units=hidden_units)

    # Set the class to index mapping and load the model's state_dict
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): The trained model to predict.

    Returns:
        probs (list): List of probabilities for the top predicted classes.
        classes (list): List of top predicted class labels.
    '''

    # Load class labels if provided
    labels_path = args.labels
    if labels_path:
        with open(labels_path, 'r') as f:
            cat_to_name = json.load(f)
    else:
        raise UserWarning('Please provide a labels file')

    topk = args.topk if args.topk else 5

    # Prepare the image for prediction
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()

    image = np.array(pil_image)

    mean = np.array(MEAN)
    std = np.array(STD)
    image = (np.transpose(image, (1, 2, 0)) - mean) / std
    image = np.transpose(image, (2, 0, 1))

    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)  # This is for VGG

    # Use GPU if available
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    image = image.to(device)

    # Perform model inference
    model.eval()
    with torch.set_grad_enabled(False):
        result = model(image).topk(topk)

    if use_gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]

    # Map class indices to class labels
    if labels_path:
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]

    return probs, classes


# Perform predictions if invoked from command line
if __name__ == '__main__':
    if args.image:
        # Load the model checkpoint
        model = load_model(args.checkpoint) if args.checkpoint else load_model()
        probs, classes = predict(args.image)
        print('Predictions and probabilities:', list(zip(classes, probs)))
