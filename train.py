import torch.cuda
from dataset import MyDataset
from torch.utils.data import DataLoader
from model.unet_model import CNN
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint, netParams, train, val
import numpy as np
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import shutil

def get_args():
    parser = argparse.ArgumentParser("""Train model for Unet""")
    parser.add_argument("--batch_size", "-b",type=int, default=8, help="batch size of dataset")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="number of epochs")
    parser.add_argument("--log_path", "-l", type=str, default="./tensorboard/", help="path to tensorboard")
    parser.add_argument("--save_path", "-s", type=str, default="./trained_model/", help="path to save model")
    parser.add_argument("--load_checkpoint", "-m", type=str, default=None, help="path to checkpoint loaded")
    args = parser.parse_args()
    return args

def train_net(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = args.batch_size
    num_epochs = args.epochs

    train_set = MyDataset(transform=False, valid=False)
    train_dataloader = DataLoader(
        dataset = train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_set = MyDataset(transform=False, valid=True)
    val_dataloader = DataLoader(
        dataset = val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(args.log_path)
    best_mIoU = 0.
    for epoch in range(start_epoch, num_epochs):
        train(train_dataloader, model, criterion, optimizer, epoch, num_epochs, writer)
        temp_mIoU = val(val_dataloader, model, writer, epoch)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(args, checkpoint)
        if temp_mIoU > best_mIoU:
            save_checkpoint(args, checkpoint, 'best.pth')
            best_mIoU = temp_mIoU

if __name__ == '__main__':
    args = get_args()
    train_net(args)