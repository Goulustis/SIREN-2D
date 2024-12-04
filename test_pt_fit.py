import numpy as np 
import cv2
import torch 
from models import *
from nn_utils import NeuralNetTorch
from pt_opt import PTOpt

SIZE = 8
# n_steps=10000
n_steps=3000

def pred2img(pred):
    pred = pred.detach().cpu().numpy()
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)
    pred = pred.reshape((SIZE,SIZE,3))
    return pred

image = cv2.imread("test.jpg")
image = cv2.resize(image, (SIZE,SIZE))
cv2.imshow("target", image)

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
relu_pe_model = ReLU_PE_Model([2,256,256,256,256,3], L=10).cuda()
siren_model = SIREN([2,8,8,8,3]).cuda()
torch_siren_def = NeuralNetTorch(input_dim=2, 
                #    hidden_dims=[256, 256, 256, 256], 
                   hidden_dims=[4,4,4,4], 
                   output_dim=3, 
                   hidden_fnc=torch.sin, 
                   out_fnc=lambda x: x,
                   device="cuda")

# initialize the optimizer
parameters = []
parameters += list(relu_pe_model.parameters())
parameters += list(siren_model.parameters())
optimizer = torch.optim.Adam(parameters, lr=1e-4)

loss_log = []
print("press ESC on any img to quit reconstruction")
for iter_idx in range(n_steps):
    relu_pe_recon = relu_pe_model(input_grid)
    siren_recon = siren_model(input_grid)

    relu_pe_loss = torch.mean((relu_pe_recon - image_torch)**2)
    siren_loss = torch.mean((siren_recon - image_torch)**2)
    total_loss = relu_pe_loss + siren_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss_log.append({
        "relu_pe_loss": relu_pe_loss.item(),
        "siren_loss": siren_loss.item()
    })

    if iter_idx % 100 == 0:
        log_str = f"{iter_idx}: relu pe loss: %f, siren loss: %f" % (
            relu_pe_loss.item(), siren_loss.item())
        print(log_str)
        relu_pe_img = pred2img(relu_pe_recon)
        siren_img = pred2img(siren_recon)
        big_img = np.concatenate([relu_pe_img, siren_img], axis=1)
        cv2.imshow("reconstructed img", big_img)
        if cv2.waitKey(10) == 27:
            break


img_np = image_torch.cpu().numpy()
grid_np = input_grid.cpu().numpy()

pt_opt = PTOpt(
    model_def=torch_siren_def,
    init_fn=torch_siren_def.init_weights,
    obj_fn=lambda thetas: -torch.mean((torch_siren_def.forward_pass(thetas, input_grid) - image_torch)**2),
    num_chains=10,
    num_iterations=n_steps,
    swap_interval=100,
    learning_rate=1e-3
)

best_weights, best_energy = pt_opt.optimize()
best_recon = pred2img(torch_siren_def.forward_pass(best_weights, input_grid))
cv2.imwrite("best_recon.jpg", best_recon)

print(f"final losses: relu_pe: {relu_pe_loss.item()}, siren: {siren_loss.item()}, pt: {best_energy}")