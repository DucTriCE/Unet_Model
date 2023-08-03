import torch
from val import SegmentationMetric, AverageMeter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import os

def save_checkpoint(args, state, filenameCheckpoint='last.pth'):
    torch.save(state, os.path.join(args.save_path, filenameCheckpoint)
)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def train(train_dataloader, model, criterion, optimizer, epoch, num_epochs, writer):
    total_batches = len(train_dataloader)
    model.train()
    train_loss = []
    progress_bar = tqdm(train_dataloader, colour='green')
    for iter, (_, input, target) in enumerate(progress_bar):
        input = input.cuda().float()
        target = target.cuda().float()
        output = model(input/255.0)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        writer.add_scalar("Train/Loss", np.mean(train_loss), total_batches*epoch + iter)
        progress_bar.set_description("Epoch: {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, np.mean(train_loss)))
        # if iter%10==0:
        #     print("Epoch: {}/{}. Iter {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, total_batches, loss))

def val(val_dataloader, model, writer, epoch):
    model.eval()
    DA = SegmentationMetric(2)
    da_mIoU_seg = AverageMeter()
    for iter, (_, input, target) in enumerate(val_dataloader):
        input = input.cuda().float()
        target = target.cuda().float()
        with torch.no_grad():
            output = model(input / 255.0)
        out_da = output
        target_da = target
        _, da_predict = torch.max(out_da, 1)
        _, da_gt = torch.max(target_da, 1)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())
        da_mIoU = DA.meanIntersectionOverUnion()
        da_mIoU_seg.update(da_mIoU, input.size(0))
    da_segment_result = da_mIoU_seg.avg
    writer.add_scalar("Val/mIoU", da_segment_result, epoch)
    return da_segment_result

