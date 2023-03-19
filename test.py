import Diffusion
import torch

model = Diffusion.UNet(device='cuda').to("cuda")
model
model.load_state_dict(torch.load("ckpt.pt"))

dif = Diffusion.Diffusion(img_size = 64, device = 'cuda')

for x in range(24):
    sample_images = dif.sample(model, n=1)
    Diffusion.save_images(sample_images, f'test_image-{x}.png')