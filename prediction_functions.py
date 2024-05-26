import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision import transforms

from PIL import Image

import numpy as np

import json

def load_checkpoint(path='home/workspace/ImageClassifier/checkpoint.pth'):

    # Loading the checkpoint dictionary,
    checkpoint = torch.load(path)
    
    # Now extracting our learning rate,
    learning_rate = checkpoint['learning_rate']
    
    trained_model = getattr(torchvision.models, checkpoint['network'])(pretrained=True)
   
    # Now loading our model attributes,
    trained_model.load_state_dict = (checkpoint['state_dict'])  
    trained_model.num_epochs = checkpoint['epochs']
    trained_model.optimizer = checkpoint['optimizer']
    trained_model.classifier = checkpoint['classifier']
    trained_model.class_to_idx = checkpoint['class_to_idx']

   

    return trained_model




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    target_size = 224
    resizing_size = 255
    norm_means = [0.485, 0.456, 0.406]
    norm_stds = [0.229, 0.224, 0.225]
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([transforms.Resize(resizing_size),
                                     transforms.CenterCrop(target_size), 
                                     transforms.ToTensor()])
    
    transformed_image = image_transforms(image).float()
    numpy_image = np.array(transformed_image)    
    
    norm_mean = np.array(norm_means)
    norm_std = np.array(norm_stds)
    numpy_image = (np.transpose(numpy_image, (1, 2, 0)) - norm_mean) / norm_std    
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
            
    return numpy_image



def predict(image_file_path, trained_model, topk = 5, use = 'gpu'):
    cuda = torch.cuda.is_available()
    if cuda and use == 'gpu':
        trained_model.cuda()
    else:
        trained_model.cpu()
    
    trained_model.eval()
    # Printing (image_file_path[0])
    image_to_predict = Image.open(image_file_path)
    image = process_image(image_to_predict)
    image = torch.from_numpy(np.array([image])).float()
    #image = Variable(image)
    
    if cuda:
        image = image.cuda()
        
    output = trained_model.forward(image)
    
    probabilities = torch.exp(output).data
    
    probs = torch.topk(probabilities, topk)[0].tolist()[0] 
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(trained_model.class_to_idx.items())):
        ind.append(list(trained_model.class_to_idx.items())[i][0])

    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return probs, label