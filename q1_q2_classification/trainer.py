from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def BCELossLogits(out, tar, wgt):
    """
    Manually calculate the weighted BCEWithLogits loss for a batch of logits and targets
    by explicitly applying the sigmoid function and considering instance weights.

    Parameters:
    out (torch.Tensor): The raw output from the network (before sigmoid).
    tar (torch.Tensor): The ground truth labels (0 or 1).
    wgt (torch.Tensor): Weights for each instance, indicating the importance or confidence.

    Returns:
    torch.Tensor: The weighted BCE loss for the batch.
    """
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(out)
    
    # Compute weighted BCE loss manually
    bce_loss = -wgt * (tar * torch.log(probabilities) + (1 - tar) * torch.log(1 - probabilities))
    
    # Return mean loss over the batch, considering only the instances with non-zero weights
    weighted_loss = torch.sum(bce_loss) / torch.sum(wgt)
    return weighted_loss



def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    # def BCELoss(output, target):
    #     """
    #     Binary cross-entropy loss function for multi-label classification.

    #     Args:
    #         - `output`: Outputs from the network
    #         - `target`: Ground truth labels, refer to voc.dataset.py
    #     Return:
    #         - `output`: Computed loss, a single floating point number.
    #     """
    #     output_prob = torch.sigmoid(output)
    #     loss_per_class = -wgt * (target * torch.log(output_prob + 1e-10) + (1 - target) * torch.log(1 - output_prob + 1e-10))
    #     return loss_per_class.sum(dim=1).mean()
    
    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            #loss = BCELoss(output, target)  # See helper function above

            # Inside your training loop
            loss = BCELossLogits(output, target, wgt)

            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
