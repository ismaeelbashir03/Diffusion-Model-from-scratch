# this is my first attempt at a Diffusion model. All the functions and classes will be in this file including the
# Unet, Diffusion forward and backwards etc. I started this project as i want to explore the architecture and math
# behind the popular dalle 2 and stable diffusion models. My version will be scaled down, as i dont have enough money
# to run a bigger model currently lol. 

#--- libraries imported ---#
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import logging
import os

#--- initalisation ---#
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

#--- classes ---#

# this is the diffusion model that holds all the diffusion functions, such as forward, backward and noise scheduling
class Diffusion():
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 256, device = 'cuda') -> None:
        
        # setting our parameters locally within the class
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # initialising a beta list of linear noise scheduling 
        # (from the origonal paper, apenAI suggest cosine scheduling for delayed full noised image)
        # i have made a cosine scheduler, but im going to test it after, the linear, since the cosine is more prone 
        # to errors/bugs, since i did it manually
        self.beta = self.linear_noise_schedule().to(device)

        # setting the alpha as the inverse of the beta
        self.alpha = 1. - self.beta

        # getting the cumulative product of the alphas, for getting noise at certain time step instead of sequential (saves time)
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

    # this is a linear noise scheduler (used in the original paper)
    def linear_noise_schedule(self):
        # using linspace, from the beta start to end in the amount of noise steps
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # cant 
    def cosine_noise_schedule(self, optimizer):
        # creating a cosine list
        beta_list = []

        # looping for each noise step
        for i in range(self.noise_steps):
            # Calculating the value of beta using the cosine annealing schedule
            t = i / self.noise_steps
            beta_t = self.beta_start + 0.5 * (self.beta_end - self.beta_start) * (1 + torch.math.cos(torch.math.pi * t))
            beta_list.append(beta_t)

        # returning the list of values
        return beta_list

    # now lets get a function to add noise to the images
    def noise_images(self, x, t):

        # we get the sqrt of the cumulative product of alpha
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]

        # then we we get the sqrt of the cumulative product of alpha minus 1
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        # then we generate an eplison value from the start image
        eps = torch.randn_like(x)

        # we can now put the function together, by doing x_0*sqrt(alpha_hat) + sqrt(1-alpha_hat)*epsilon
        # we return this with the epsilon used. (the epsilon is used to randomly sample noise)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    # helper function to randomly sample timesteps
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # lets finally make our sampling function, we take in our Unet, and the number of images.
    # this si trained with batches, so functions are done with n dimensionality added throughout the code
    def sample(self, model, n):

        # we set the model to evaluate
        model.eval()

        # we then use torch's nograd function so we dont get gradients from the following code
        with torch.no_grad():

            # we get a random rgb (3 channels) image (of noise) from a normal distribution, n is the number of batches
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # for each time step, going backwards
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

                # we get a tensor for the current timestep
                t = (torch.ones(n) * i).long().to(self.device)

                # we now predict the noise that was added
                predicted_noise = model(x, t)

                # we now get the alpha of the timestep along witht he cumulative product of alpha and the beta
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # if the timestep is not 0 (at 'assumed' original image)
                if i > 1:
                    # we calculate the noise to add to the image
                    noise = torch.randn_like(x)

                # if not, we are at the final image so we set noise to zero, no point in adding noise
                else:
                    noise = torch.zeros_like(x)

                # here we subtract a fraction of the noise from the image, to attempt to go back one timestep, we dont subtract all of the noise
                # as the papers found that denoising step by step produced better results. This is then mulitplied by 1/sqrt(alpha) and then additional
                # noise is added (scaled by beta), since we want to change the image at each time step (get different images from training), becuase if we dont
                # do this we risk reprodcuing the same images as training. (basically adds 'creativity')
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            # we can now go back to training mode
            model.train()

            # we now clip the output to be between -1 and 1, then to 0 and 1
            x = (x.clamp(-1, 1) + 1) / 2

            # then we multiply by 255 to get rgb pixel values
            x = (x * 255).type(torch.uint8)

            return x

# this is the Unet model class, where we use a nueral network to learn the noise added to an image
# (this is a child of the pytorch module)
class UNet(nn.Module):

    # this calss takes in the number of channels (3 for rgb)
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256, device = 'cuda') -> None:

        # we call the super constructor for the pytorch module
        super().__init__()

        # lets set our variables local to the class
        self.device = device
        self.time_dim = time_dim

        # here we define the architecure of the UNet model, this consists of 
        # a downsampling block then a bottleneck block then a final upsampling block

        #--- Down sample block ---#

        # going from 64x64 size to 32, to 16, to 8 (we reduce dimensionality in the self attention output)

        self.inc = DoubleConv(c_in, 64)     # input layer that is a double conv layer, takes in image channel len by size
        self.down1 = Down(64, 128)          # down layer is a down sampling layer, takes in the size and output channel for the image
        self.sa1 = SelfAttention(128, 32)   # this is a self attention layer, similiar to the transformer model, takes in the channel and size of image
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        #--- Bottleneck block ---#
        # here have a bunch of convolutional layers
        self.bot1 = DoubleConv(256, 512)    # double conv layer, is two conv2d layers
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        #--- Up sample block ---#
        # here we do the opposite of the down sample layer
        self.up1 = Up(512, 128)              # up layer is a upsampling layer, takes in the channel and size of the image
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1) # output layer is a simple conv2d layer that outputs a 64x64 image with 3 channels (rgb)

    # this is a function for the positional encoding of the image, (i.e. when we split the image into vectors 
    # for self attention we need the position of that vector with respect to the image for extra information)
    # here we use a common timestep embedding formula used to get the positional encoding in most papers using sin and cosine.
    def pos_encoding(self, t, channels):

        # here we get the inverse frequency and create a tensor of shape channels//2
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )

        # we then get the positional encoding for cosine and sin using our inverse frequency and the timestep
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        # we then concatonate the positional encodings to the get the final encoding
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)

        return pos_enc

    # here we define the forward pass
    def forward(self, x, t):

        # we first get the positional encoding of the tensors
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # we then do a forward pass as normal, where the self attention takes the positional encoding of course.

        # forward through down sample
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # forward through bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # forward through up sample
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        return output

# here we define our double convolutional layer
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()

        # we first get the residual bool to check if we are doing residuals
        self.residual = residual

        # # then we check if are not using mid channels (changing channels in between pass)
        if not mid_channels:
            # if we dont we set the mid channels to the output channels
            mid_channels = out_channels

        # here we use a sequrntial model to define two conv2d layers (without bias, using a 3x3 filter), with group normalisation and Gelu (tilted relu)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )
    
    # here we define the forward pass
    def forward(self, x):

        # if we are using residuals, we add the residual connection, by adding the input to the output
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        # if not we just return the forward pass as normal
        else:
            return self.double_conv(x)

# here we define our down sample layer
class Down(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # we define a sequential model that does a max pooling to half the input and uses two double convolutional layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        # then we define the embedding layer, which linearly transforms the embedding
        # dimension of the positional embedding to the number of channels for the output
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            )
        )

    # here we define the forward pass
    def forward(self, x, t):

        # we go through the maxpooling layer
        x = self.maxpool_conv(x)

        # then we go through the embedding layer, using the positional embedding passed in
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        # we can then return the residual of this (by adding the input)
        return x + emb

# this is the up sampling block, similiar to the down sample
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # we define a upsample layer, witha  factor of 2, opposite of the down sample division by 2
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # we then define a conv layer consisting of 2 double convolutional layers, exaclty the same as down sample
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # then we also do a embedding layer to transform the positional encoding dimension to the channel size
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    # here we define the forward pass
    def forward(self, x, skip_x, t):

        # we go through the up sample (with residual) and the conv layer
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        # then we pass through the embedding layer and make a residual connection
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

# here we define a self attention layer, almost identical to the transformer model previously
class SelfAttention(nn.Module):

    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        # we locally define our variables
        self.channels = channels
        self.size = size

        # then we use pytorches multihead attention layer, (for details look at my transformer model from scratch)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        # we then do layer normalisation
        self.ln = nn.LayerNorm([channels])

        # then we do a linear transformation after using another layer nomralisation
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    # here we define the forward pass
    def forward(self, x):

        # we need to reshape the input so we can make them the right shape for the self attention
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)

        # we go through our layer nomralisation
        x_ln = self.ln(x)

        # then we pass through our self attention , with a residual connection
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x

        # we the  go through our feed forward linear transformation, with a residual connection
        attention_value = self.ff_self(attention_value) + attention_value

        # we revert the shape back before we return the ouput
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

# here we define a function to plot the images that have been generated
def plot_images(images):

    # we set the figure size
    plt.figure(figsize=(32, 32))

    # we then show each image of the images
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())

    plt.show()

# here we define a function to save the images
def save_images(images, path, **kwargs):

    # we make a grid of the image
    grid = torchvision.utils.make_grid(images, **kwargs)

    # we then permute the image, then convert it to an array
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()

    # we then load and save the images with PIL
    im = Image.fromarray(ndarr)
    im.save(path)

# here we have a function to transform our mdata to fit the model, this ressizes the images and converts them to a tensor
def get_data(args):

    # this the transformer that resizes the images, crops them, and converts them to a tensor and normalises them to 0 and 1
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # we then get the dataset and use the transformer function to apply the transformation to each image
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    
    # then we load the data into batches and shuffle them
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # we then the return the data
    return dataloader

# this is the function to train the model, the formula taken from the paper
def train(args):

    # we setup our logging, device, data, and model
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # we then setup the diffusion model
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # we finish setting up the logger
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    # for epoch in our training step
    for epoch in range(args.epochs):

        # we log the epoch
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        # for each image
        for i, (images, _) in enumerate(pbar):

            # we set the device for computation
            images = images.to(device)

            # we get a sample timestep (random) for the image
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # we then get the noised image and the noise used
            x_t, noise = diffusion.noise_images(images, t)

            # we then predict the noise of the image with the unet
            predicted_noise = model(x_t, t)

            # then we calculate the loss, using mean squared error (used in paper)
            loss = mse(noise, predicted_noise)

            # we set the optimizer to zero gradients initially
            optimizer.zero_grad()

            # we do our backward step
            loss.backward()

            # we then optimize using our gradients
            optimizer.step()

            # we then log our loss
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        
        # we then sample some images after training
        sampled_images = diffusion.sample(model, n=images.shape[0])

        # we save hte images and the model
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

# here we define a function to setup the logging
def setup_logging(run_name):

    # we make folders for the model and results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # we make sub folders for the runs
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

# here we define a function to start training our data
def launch():

    # we use argument parsing to ass in our data to the train loop
    import argparse

    # we setup the parser
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # then we load the parsing data
    args.run_name = "DDPM_Uncondtional_1"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"data/"
    args.device = "cuda"
    args.lr = 3e-4

    # we pass the data in
    train(args)

# we launch when this file is ran
if __name__ == '__main__':
    launch()