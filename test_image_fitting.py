import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch 
from torch import nn
from models import *
import os.path as osp
import os
import copy
from utils import calc_psnr

os.makedirs("img_cache", exist_ok=True)


SIZE = 256

def pred2img(pred):
    pred = pred.detach().cpu().numpy()
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)
    pred = pred.reshape((SIZE,SIZE,3))
    return pred

image = cv2.imread("cropped_00.png")
image = cv2.resize(image, (SIZE,SIZE))
gt_image = copy.deepcopy(image)


# generate the target data of training
image_torch = torch.from_numpy(image.astype(np.float32)/255.0)
image_torch = image_torch.view(-1,3).cuda()

# generate input image grids
xs, ys = np.meshgrid(np.linspace(-1,1,SIZE), np.linspace(-1,1,SIZE))
# xs, ys = np.meshgrid(np.linspace(0,511,512), np.linspace(0,511,512))
xs = torch.from_numpy(xs.astype(np.float32)).view(SIZE,SIZE,1)
ys = torch.from_numpy(ys.astype(np.float32)).view(SIZE,SIZE,1)
input_grid = torch.cat([ys, xs],dim=2)
input_grid = input_grid.view(-1,2)
input_grid = input_grid.cuda()

# initialize the models
# relu_model = ReLU_Model([2,256,256,256,256,3]).cuda()
relu_pe_model = ReLU_PE_Model([2,256,256,256,256,3], L=10).cuda()
# siren_model = SIREN([2,256,256,256,256,3]).cuda()

bottle_model = nn.Sequential(
    copy.deepcopy(relu_pe_model),
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,3),
    # nn.Sigmoid()
).cuda()

# initialize the optimizer
parameters = []

parameters += list(relu_pe_model.parameters())
parameters += list(bottle_model.parameters())

optimizer = torch.optim.Adam(parameters, lr=1e-4)

loss_log = []
print("press ESC on any img to quit reconstruction")
for iter_idx in range(10000):

    relu_pe_recon = relu_pe_model(input_grid)
    bottle_recon = bottle_model(input_grid)

    relu_pe_loss = torch.mean((relu_pe_recon - image_torch)**2)
    bottle_loss = torch.mean((bottle_recon - image_torch)**2)

    total_loss =  relu_pe_loss + bottle_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss_log.append({
        "relu_pe_loss": relu_pe_loss.item()
    })

    if iter_idx % 100 == 0:
        # log_str = "iter: %f, relu pe loss: %f, bottle loss: %f" % (
        #     iter_idx, relu_pe_loss.item(), bottle_loss.item())
        # print(log_str)
        
        relu_pe_img = pred2img(relu_pe_recon)
        bottle_img = pred2img(bottle_recon)

        log_str = f"iter:{int(iter_idx)}, relu_pe_psnr: {calc_psnr(relu_pe_recon, gt_image)}, relu_gamma_psnr: {calc_psnr(bottle_img, gt_image)}"
        
        big_img = np.concatenate([ relu_pe_img, bottle_img], axis=1)
        # big_img = relu_pe_img
        cv2.imwrite(osp.join("img_cache", f"{str(iter_idx//100).zfill(5)}.png"), big_img)

# relu_losses = [i["relu_loss"] for i in loss_log]
# relu_pe_losses = [i["relu_pe_loss"] for i in loss_log]
# siren_losses = [i["siren_loss"] for i in loss_log]
# step_num = len(loss_log)
# plt.plot(range(step_num), relu_losses, label='relu_losses')
# plt.plot(range(step_num), relu_pe_losses, label='relu_pe_losses')
# plt.plot(range(step_num), siren_losses, label='siren_losses')
# plt.legend(loc='upper right')
# plt.show()
