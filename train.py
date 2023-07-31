import torch.cuda
from dataset import MyDataset
from torch.utils.data import DataLoader
from model.unet_model import CNN
import torch.nn as nn
import torch.optim as optim
from utils import save_checkpoint, netParams, train, val
import numpy as np

def train_net():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 8
    num_epochs = 50

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

    for epoch in range(num_epochs):
        model_file_name = 'pretrained/model_{}.pth'.format(epoch)
        train(train_dataloader, model, criterion, optimizer)
        print(val(val_dataloader, model))
        torch.save(model.state_dict(), model_file_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })


if __name__ == '__main__':
    train_net()