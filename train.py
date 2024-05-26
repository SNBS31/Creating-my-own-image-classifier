import argparse

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import trainer_functions
import prediction_functions

argument_parser = argparse.ArgumentParser(description = 'Train.py') 

argument_parser.add_argument('data_dir', nargs = '*', action = 'store', default = './flowers/')
argument_parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu')
argument_parser.add_argument('--save_dir', dest = 'save_dir', action='store', default = './checkpoint.pth')
argument_parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.01)
#argument_parser.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.5)
argument_parser.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 1)
argument_parser.add_argument('--arch', dest = 'arch', action = 'store', default = 'vgg16', type = str)
argument_parser.add_argument('--hidden_units', type = int, dest = 'hidden_units', action = 'store', default = 120)

argument_parser = argument_parser.parse_args()
data_dir = argument_parser.data_dir
path = argument_parser.save_dir
lr = argument_parser.learning_rate
base_architecture = argument_parser.arch
#dropout_probability = parser.dropout
intermediate_features = argument_parser.hidden_units
processing_unit = argument_parser.gpu
num_epochs = argument_parser.epochs


training_data_loader, validation_loader, testing_data_loader, train_data = trainer_functions.fetch_image_data(data_dir)
trained_model, error_measure, optimizer = trainer_functions.build_model(base_architecture, intermediate_features, lr, processing_unit)
trainer_functions.train_model(trained_model, error_measure, optimizer, validation_loader, training_data_loader, processing_unit, num_epochs)
trainer_functions.save_checkpoint(trained_model, optimizer, train_data, base_architecture, path, intermediate_features, lr, num_epochs)
print("Done") 