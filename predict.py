import json

import numpy as np

import argparse

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

import trainer_functions
import prediction_functions

# Now creating the argument parser with descriptive description,
argument_parser = argparse.ArgumentParser(description="Program for image prediction using a trained model")

# Add arguments for required inputs
argument_parser.add_argument(
    "img_path", 
    default="/home/workspace/ImageClassifier/flowers/test/16/image_06670.jpg", 
    type=str, 
    nargs="*"
)
argument_parser.add_argument("model_checkpoint", default="/home/workspace/ImageClassifier/checkpoint.pth", type=str, nargs="*")

# Add optional arguments for customization
argument_parser.add_argument("--top_predictions", default=5, dest="top_k", type=int)
argument_parser.add_argument("--category_mapping", dest="category_names", default="cat_to_name.json")
argument_parser.add_argument("--processor", default="gpu", dest="gpu")

# Parse the provided arguments
parsed_args = argument_parser.parse_args()

# Extract arguments for clarity
image_file_path = parsed_args.img_path
model_file_path = parsed_args.model_checkpoint
topk = parsed_args.top_k
processing_unit = parsed_args.gpu

# Load necessary data for prediction
data_loaders = trainer_functions.fetch_image_data() 

# Load the trained model from checkpoint
trained_model = prediction_functions.load_checkpoint(model_file_path)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    # What about process_image?

probabilities = prediction_functions.predict(image_file_path, trained_model, topk, processing_unit)

labels = [cat_to_name[index] for index in probabilities[1]]
probability = np.array(probabilities[0])

index = 0
while index < topk:
    print("There is a {} % chance that this photo shows a {}.".format(probability[index] * 100, labels[index]))
    index += 1

print("Done")