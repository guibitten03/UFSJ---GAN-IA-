import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataset import data
from model import *

# === COMMAND LINE PARSER == #
parser = argparse.ArgumentParser()

parser.add_argument("-e", dest="num_epochs", type=int, default=10, help="Num of train epochs")

args = parser.parse_args()


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
        
    # numerically stable loss
    '''
    BCEWithLogisticLoss:
    
    Essa perda combina uma camada Sigmoid e a BCELoss em uma única classe. Essa versão é mais numericamente estável do que usar 
    uma Sigmoid simples seguida por uma BCELoss, pois, ao combinar as operações em uma única camada, aproveitamos o truque de 
    log-sum-exp para estabilidade numérica.
    '''
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def train(dataset_loader, D, G, g_optimizer, d_optmizer, epochs):

    # training hyperparams
    num_epochs = 50

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 400

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    D.train()
    G.train()
    for epoch in range(epochs):
        
        for batch_i, (real_images, _) in enumerate(dataset_loader):
                    
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
                z = np.random.uniform(-1, 1, size=(batch_size, z_size))
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
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
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
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item()))

        
        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))
        
        # generate and save sample, fake images
        G.eval() # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to train mode


    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
        
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("results/20e_training_loss.png")
    
    # randomly generated, new latent vectors
    sample_size=16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()

    G.eval() # eval mode
    # generated samples
    rand_images = G(rand_z)
    
    # helper function for viewing a list of passed in sample images
    def view_samples(epoch, samples):
        fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch]):
            img = img.detach()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            plt.savefig(f"results/20e_samples_{epochs}.png")

    # 0 indicates the first set of samples in the passed in list
    # and we only have one batch of samples, here
    view_samples(10, [rand_images])


if __name__ == "__main__":
    dataset = data()
    
    # Hiperparametros do Discriminador

    # Size of input image to discriminator (28*28)
    input_size = 784
    # Size of discriminator output (real or fake)
    d_output_size = 1
    # Size of last hidden layer in the discriminator
    d_hidden_size = 256

    # Hiperparametros do Gerador

    # Size of latent vector to give to generator
    z_size = 64
    # Size of discriminator output (generated image)
    g_output_size = 784
    # Size of first hidden layer in the generator
    g_hidden_size = 256
    
    D = Discriminator(input_size, d_hidden_size, d_output_size)
    G = Generator(z_size, g_hidden_size, g_output_size)
    

    # Criação de otimizadores pro Discriminador e Gerador
    d_optimizer = optim.Adam(D.parameters(),  lr=0.0002, weight_decay=0.00001)
    g_optimizer = optim.Adam(G.parameters(),  lr=0.0002, weight_decay=0.00001)
    
    train(data.dataset_loader, D, G, g_optimizer, d_optimizer, args.num_epochs)