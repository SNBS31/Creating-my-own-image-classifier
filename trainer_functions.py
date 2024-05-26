import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

import time
import os

# Let's start by importing our required datasets in
def fetch_image_data(data_dir="./flowers"):

   data_dir = str(data_dir).strip('[]').strip("'")
   train_dir = os.path.join(data_dir, "train")
   valid_dir = os.path.join(data_dir, "valid")
   test_dir = os.path.join(data_dir, "test")

   # TODO: Define your transforms for the training, validation, and testing sets

   # For all three sets you'll need to normalize the means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]
   # data_transforms =

   target_size = 224
   resizing_size = 255
   norm_means = [0.485, 0.456, 0.406]
   norm_stds = [0.229, 0.224, 0.225]

   training_transforms = transforms.Compose([
       transforms.RandomResizedCrop(target_size),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize(norm_means, norm_stds)
   ])

   validation_transforms = transforms.Compose([
       transforms.RandomResizedCrop(target_size),
       transforms.ToTensor(),
       transforms.Normalize(norm_means, norm_stds)
   ])

   testing_transforms = transforms.Compose([
       transforms.Resize(resizing_size),  # Why 255 pixels?
       transforms.CenterCrop(target_size),
       transforms.ToTensor(),
       transforms.Normalize(norm_means, norm_stds)
   ])

   # TODO: Load the datasets with ImageFolder

   batch_size = 60

   # Data loading
   training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
   validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
   test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

   dataloaders = [
       torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True),
       torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True),
       torch.utils.data.DataLoader(test_data)
   ]

   return dataloaders[0], dataloaders[1], dataloaders[2], training_data

# Now let's build a model
def build_model(base_architecture='vgg16', intermediate_features=1024, learning_rate=0.01, computational_device='gpu'):

   if torch.cuda.is_available() and computational_device == 'gpu':
       model_device = torch.device('cuda')
   else:
       model_device = torch.device('cpu')

   if base_architecture == 'vgg16':
       trained_model = models.vgg16(pretrained=True).to(model_device)
   elif base_architecture == 'densenet121':
       trained_model = models.densenet121(pretrained=True).to(model_device)
   elif base_architecture == 'alexnet':
       trained_model = models.alexnet(pretrained=True).to(model_device)
   else:
       print(f"Im sorry but {base_architecture} isn't a valid model. Did you mean vgg16, densenet121 or alexnet?")
       return None, None, None

    # to avoid backdroping of the parameters
   for param in trained_model.parameters():
      param.requires_grad = False

   initial_features = 25088
   classes_count = 102
   dropout_rate = 0.5
    
   classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(initial_features, intermediate_features)),
                                ('drop', nn.Dropout(p = dropout_rate)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(intermediate_features, classes_count)),
                                ('output', nn.LogSoftmax(dim = 1))
                              ]))    


   trained_model.classifier = classifier

   error_measure = nn.NLLLoss()
   optimizer = optim.SGD(trained_model.classifier.parameters(), lr=learning_rate)
        
   trained_model.to(model_device)

   return trained_model, error_measure, optimizer


# Now let's train our model
def train_model(trained_model, error_measure, optimizer, validation_data_loader, training_data_loader, hardware_preference='gpu', num_epochs=1):

    validation = True
    
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(model_device)

    start_timestamp = time.time()
    print('Commencement of training phase')
    epoch_loss = 0
    training_loss = 0
    for epoch_index in range(num_epochs):
        for images, labels in training_data_loader:
            images, labels = images.to(model_device), labels.to(model_device)

            optimizer.zero_grad()

            model_predictions = trained_model(images)
            loss = error_measure(model_predictions, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print('Epoch {} of {} /// Training loss {:.3f} ///'.format(epoch_index + 1, num_epochs, epoch_loss / len(training_data_loader)))

        if validation:
            validation_loss = 0
            validation_accuracy = 0
            trained_model.eval()
            with torch.no_grad():
                for images, labels in validation_data_loader:
                    images, labels = images.to(model_device), labels.to(model_device)

                    model_predictions = trained_model(images)
                    loss = error_measure(model_predictions, labels)

                    validation_loss += loss.item()

                    probabilities = torch.exp(model_predictions)
                    top_probability, top_class = probabilities.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            trained_model.train()

            print("Validation loss {:.3f} /// Validation accuracy {:.3f} ///".format(validation_loss / len(validation_data_loader), validation_accuracy / len(validation_data_loader)))

    end_timestamp = time.time()
    print('Training phase concludes')

    total_training_time = end_timestamp - start_timestamp
    print('Total training duration: {:.0f}m {:.0f}s'.format(total_training_time / 60, total_training_time % 60))

    return trained_model


def save_checkpoint(trained_model, optimizer, train_data, base_architecture = 'vgg16', path = 'checkpoint.pth', initial_features = 25088, intermediate_features = 1024, classes_count = 102, lr = 0.01, num_epochs = 1, batch_size = 64):
    trained_model.class_to_idx = train_data.class_to_idx

    checkpoint = {'network': 'vgg16',
                'input': initial_features,
                'output': classes_count,
                'learning_rate': lr,       
                'batch_size': batch_size,
                'classifier' : trained_model.classifier,
                'epochs': num_epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': trained_model.state_dict(),
                'class_to_idx': trained_model.class_to_idx}
    
    torch.save(checkpoint, path)
    
