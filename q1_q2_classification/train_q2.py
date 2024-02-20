import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
from torchvision.models import ResNet18_Weights # added

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        #self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        weights = ResNet18_Weights.IMAGENET1K_V1  # Correct way to specify weights
        self.resnet = torchvision.models.resnet18(weights=weights)
        ##################################################################
        # TODODefine a FC layer here to process the features
        ##################################################################

        #self.resnet18 = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Define the new fully connected layer
        self.fc = nn.Linear(1000, num_classes)

        self.transforms = weights.transforms(antialias=True)
        #self.transforms = weights.transforms()
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        with torch.no_grad():
            x = self.transforms(x)
        
        # Pass input through ResNet18
        x = self.resnet(x)
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        
        # Pass the features through the new fully connected layer
        y_pred = self.fc(x)
        
        return y_pred
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    ##################################################################
    args = ARGS(
        epochs=15,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr= 0.001,
        batch_size= 64,
        step_size= 1,
        gamma= 0.7
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
