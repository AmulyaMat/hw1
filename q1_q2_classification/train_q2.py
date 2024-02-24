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

class ResNet_TSNE(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet = torchvision.models.resnet18(weights=weights)

        self.flat_dim = 1000
        self.extra_fc = nn.Linear(self.flat_dim, num_classes)
        

    def forward(self, x):
      
      x = self.resnet(x)  # The output here is already flat, with a size of 512
      out = self.extra_fc(x)  # Additional FC layer

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
        epochs=15,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr= 0.00001,
        batch_size= 32,
        step_size= 5,
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

    model = ResNet_TSNE(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    torch.save({
        'ResNet_TSNE2_state_dict': model.state_dict(),
    }, 'ResNet_TSNE2_checkpoint.pth')

    print('test map:', test_map)
