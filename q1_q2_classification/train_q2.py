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

# class ResNet(nn.Module):
#     def __init__(self, num_classes) -> None:
#         super().__init__()

#         weights = ResNet18_Weights.IMAGENET1K_V1
#         self.resnet = torchvision.models.resnet18(weights=weights)

#         # # Disable gradient updates for all layers.
#         # for param in self.resnet.parameters():
#         #     param.requires_grad = False

#         # # Get the number of features of the original final FC layer.
#         # num_features = self.resnet.fc.in_features

#         # # Replace the original ResNet's final FC layer with a new one that outputs features for the extra_fc layer.
#         # # Assuming `extra_fc` outputs features directly usable by the classifier.
#         # self.resnet.fc = nn.Identity()

#         # Define the additional FC layer that takes in features from the modified ResNet and outputs num_classes features.
#         self.extra_fc = nn.Linear(1000, num_classes)

#         # Final classifier that maps the output of extra_fc to the desired number of classes.
#         # If extra_fc already performs the final classification, this layer might not be necessary.
#         self.flat_dim = 1000
#         ##################################################################
#         #                          END OF YOUR CODE                      #
#         ##################################################################
        

#     def forward(self, x):
#         ##################################################################
#         # TODO: Return raw outputs here
#         # ##################################################################
#         # # with torch.no_grad():
#         # #     x = self.transforms(x)
        
#         # # Pass input through ResNet18
#         # x = self.resnet(x)
        
#         # # Flatten the output for the fully connected layer
#         # x = torch.flatten(x, 1)
        
#         # # Pass the features through the new fully connected layer
#         # y_pred = self.fc(x)
        
#         # return y_pred


#         N = x.size(0)
#         x = self.resnet(x)
#         flat_x = x.view(N, self.flat_dim)
#         out = self.extra_fc(flat_x)
#         return out


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        # Load pre-trained ResNet-18 model
        self.resnet = torchvision.models.resnet18(weights=weights)
        
        # Freeze all pre-trained layers (optional)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Get the number of features in the input of the original final FC layer
        num_features = self.resnet.fc.in_features

        # Replace the original ResNet's final FC layer with an identity function, 
        # effectively removing it from affecting the forward pass
        self.resnet.fc = nn.Identity()

        # Define the additional FC layer(s)
        self.additional_fc = nn.Linear(num_features, 512)  # Example intermediate layer
        self.final_fc = nn.Linear(512, num_classes)  # Final classification layer

    def forward(self, x):
        # Extract features using the modified ResNet model (without its original final layer)
        features = self.resnet(x)
        
        # Pass the extracted features through the additional FC layers
        additional_features = self.additional_fc(features)
        out = self.final_fc(additional_features)
        
        return out




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
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr= 0.0003,
        batch_size= 64,
        step_size= 2,
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
    torch.save(model, 'resnet_model.pth')
    print('test map:', test_map)
