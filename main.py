from utils.config import Configuration
from model.discriminator import Discriminator
from model.generator import Generator
from metrics.loss import *
from dataset.dataset import DataLoader

import torch
import torch.optim as optim
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt



def train(opt, data_loader, D, G, d_optimizer, g_optimizer):
    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []
    
    # train the network
    D.train()
    G.train()
    for epoch in range(opt.num_epochs):
        
        for batch_i, (real_images, _) in enumerate(data_loader):
                    
            batch_size = real_images.size(0)
            
            ## Important rescaling step ## 
            real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
            
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            
            d_optimizer.zero_grad()
            
            # 1. Train with real images

            # Compute the discriminator losses on real images 
            # smooth the real labels
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)
            
            # 2. Train with fake images
            
            # Generate fake images
            # gradients don't have to flow during this step
            with torch.no_grad():
                z = np.random.uniform(-1, 1, size=(batch_size, opt.z_size))
                z = torch.from_numpy(z).float()
                fake_images = G(z)
            
            # Compute the discriminator losses on fake images        
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)
            
            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            
            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            
            # 1. Train with fake images and flipped labels
            
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, opt.z_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake) # use real loss to flip labels
            
            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % opt.print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, opt.num_epochs, d_loss.item(), g_loss.item()))

        
        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(opt.fixed_z)
        samples.append(samples_z)
        G.train() # back to train mode


    # Save training generator samples
    with open('checkpoints/train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
        
        
def test(opt, G, save_path):
    # randomly generated, new latent vectors
    sample_size=16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, opt.z_size))
    rand_z = torch.from_numpy(rand_z).float()

    G.eval() # eval mode
    # generated samples
    rand_images = G(rand_z)
    
    with open('checkpoints/train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)
    
    def view_samples(epoch, samples):
        fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch]):
            img = img.detach()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    view_samples(0, [rand_images])
    plt.savefig(save_path)
    


if __name__ == "__main__":
    data = DataLoader()
    
    opt = Configuration()
    
    opt.num_epochs = 10
    opt.print_every = 400
    
    opt.sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(opt.sample_size, opt.z_size))
    opt.fixed_z = torch.from_numpy(fixed_z).float()
    
    # instantiate discriminator and generator
    D = Discriminator(opt.input_size, opt.d_hidden_size, opt.d_output_size)
    G = Generator(opt.z_size, opt.g_hidden_size, opt.g_output_size)
    
    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(),  lr=0.0002, weight_decay=0.00001)
    g_optimizer = optim.Adam(G.parameters(),  lr=0.0002, weight_decay=0.00001)
    
    train(opt, data.dataset_loader, D, G, d_optimizer, g_optimizer)
    test(opt, G, "results/output_100e.png")