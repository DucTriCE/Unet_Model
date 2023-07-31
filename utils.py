import torch
from val import SegmentationMetric, AverageMeter
import numpy as np

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def train(train_dataloader, model, criterion, optimizer):
    total_batches = len(train_dataloader)
    model.train()
    for iter, (_, input, target) in enumerate(train_dataloader):
        # print(target.shape)
        output = model(input/255.0)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if iter%10==0:
        #     print("Epoch: {}/{}. Iter {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, total_batches, loss))

def val(val_dataloader, model):
    model.eval()
    DA = SegmentationMetric(2)
    da_mIoU_seg = AverageMeter()
    for iter, (_, input, target) in enumerate(val_dataloader):
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
    return da_segment_result

