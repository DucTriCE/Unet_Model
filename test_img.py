import numpy as np
import shutil
import os
import torch
from model.unet_model import CNN
import cv2

def Run(model, img):
    img = cv2.resize(img, (160, 80))                #Width=160, Height=80
    img_rs = img.copy()
    #print(img_rs.shape)                            #(Height, width, channel)
    img = img[:, :, ::-1].transpose(2, 0, 1)        #[:, :, ::-1] reverse fromRGBtoBGR (Orinal color of cv2)
    # print(img.shape)                              #After transpose: (channel, Height, width)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img/255.0
    with torch.no_grad():
        img_out = model(img)
    x0 = img_out[0]
    _, da_predict = torch.max(x0, 0)
    # print(da_predict.size())
    DA = da_predict.byte().data.numpy() * 255       #Background 0, lane: 255
    # print(DA.shape)
    img_rs[DA > 100] = [255, 0, 0]                  #DA>100:
    return img_rs

model = CNN()
checkpoint = torch.load('trained_model/best.pth')
model.load_state_dict(checkpoint["state_dict"])

model.eval()
image_list = os.listdir('test_images')
shutil.rmtree('results')
os.mkdir('results')
for i, imgName in enumerate(image_list):
    img = cv2.imread(os.path.join('test_images', imgName))
    img = Run(model, img)
    cv2.imwrite(os.path.join('results', imgName), img)