import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt


from model import *
from dataset import data

# === COMMAND LINE PARSER == #
parser = argparse.ArgumentParser()

parser.add_argument("-e", dest="num_epochs", type=int, default=10, help="Num of train epochs")

args = parser.parse_args()


def train(device, dataloader, gen_model, disc_model, gen_optimizer, disc_optimizer, epochs):
    criterion = nn.BCELoss()
    
    gen_model.train()
    disc_model.train()
    
    # deifne labels for fake images and real images for the discriminator
    fake_label = 0
    real_label = 1

    # define a fixed noise 
    fixed_noise = torch.randn(64, noise_channels, 1, 1).to(device)

    # make the writers for tensorboard
    writer_real = SummaryWriter(f"runs/fashion/test_real")
    writer_fake = SummaryWriter(f"runs/fashion/test_fake")

    # define a step
    step = 0

    print("Start training...")

    # loop over all epochs and all data
    d_loss, g_loss = [], []
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # set the data to cuda
            data = data.to(device)

            # get the batch size 
            batch_size = data.shape[0]

            # Train the discriminator model on real data
            disc_model.zero_grad()
            label = (torch.ones(batch_size) * 0.9).to(device)
            output = disc_model(data).reshape(-1)
            real_disc_loss = criterion(output, label)
            d_x = output.mean().item()

            # train the disc model on fake (generated) data
            noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
            fake = gen_model(noise)
            label = (torch.ones(batch_size) * 0.1).to(device)
            output = disc_model(fake.detach()).reshape(-1)
            fake_disc_loss = criterion(output, label)

            # calculate the final discriminator loss
            disc_loss = real_disc_loss + fake_disc_loss

            # apply the optimizer and gradient
            disc_loss.backward()
            disc_optimizer.step()

            # train the generator model
            gen_model.zero_grad()
            label = torch.ones(batch_size).to(device)
            output = disc_model(fake).reshape(-1)
            gen_loss = criterion(output, label)
            # apply the optimizer and gradient
            gen_loss.backward()
            gen_optimizer.step()
            
            
        d_loss.append(disc_loss.cpu().item())
        g_loss.append(gen_loss.cpu().item())            
    
            
        print(
                f"Epoch: {epoch} ===== Disc loss: {disc_loss:.4f} ===== Gen loss: {gen_loss:.4f}"
        )
        if epoch % 10 == 0:
                # print everything

                        ## test the model
            with torch.no_grad():
                fake = torch.randn(16, noise_channels, 1, 1).to(device)
                            
                imgs = gen_model(fake).cpu()
                fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
                for ax, img in zip(axes.flatten(), imgs):
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    im = ax.imshow(img.reshape((64,64)), cmap='Greys_r')
                        
                plt.savefig(f"results/100e_samples_{epoch}.png")
                plt.close()
                        
    fig, ax = plt.subplots()
    plt.plot(d_loss, label='Discriminator')
    plt.plot(g_loss, label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("results/100e_training_loss.png")
    plt.close()
    


if __name__ == "__main__":
    image_channels = 1
    noise_channels = 256
    gen_features = 64
    disc_features = 64
    
    device = torch.device("cuda")
    
    
    # load models
    gen_model  = Generator(noise_channels, image_channels, gen_features).to(device)
    disc_model = Discriminator(image_channels, disc_features).to(device)
    
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.0005, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=0.0005, betas=(0.5, 0.999)) 
    
    train(device, data.dataloader, gen_model, disc_model, gen_optimizer, disc_optimizer, args.num_epochs)
    