import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

import utils
from models import Generator, Discriminator
import models

# Global Settings
BATCH_SIZE = 16
LATENT_DIM = 128
CONV_DIM = 64
D_NORM_LAYER = "SN" # switch to "BN" to run baseline models
SAVE_FOLDER = "results/"
MODEL_SAVE_FOLDER = "pretrained_models/"

# Optimizer Settings
LR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999

# Training Settings
NUM_EPOCHS = 2

def real_loss(x, smooth=False):
    # Get the number of samples
    batch_size = x.shape[0]

    # Label Smoothing
    if (smooth):
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # We need to squeeze because we are comparing tensor (batch_size, 1) and (batch_size)
    return criterion(x.squeeze(), labels)

def fake_loss(x, smooth=False):
    # Get the number of samples
    batch_size = x.shape[0]

    # Label Smoothing
    if (smooth):
        labels = torch.zeros(batch_size) + 0.1
    else:
        labels = torch.zeros(batch_size)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # We need to squeeze because we are comparing tensor (batch_size, 1) and (batch_size)
    return criterion(x.squeeze(), labels)

def main():
    # Seeds
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.SVHN('data', split="train", transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Networks
    d = Discriminator(conv_dim=CONV_DIM, norm_layer=D_NORM_LAYER).to(device)
    g = Generator(latent_dim=LATENT_DIM, conv_dim=CONV_DIM).to(device)

    # Apply Weight Initialization
    d.apply(models.weights_init_xavier_normal)
    g.apply(models.weights_init_xavier_normal)

    # Optimizer Settings
    d_optim = optim.Adam(d.parameters(), lr=LR, betas=[BETA_1, BETA_2])
    g_optim = optim.Adam(g.parameters(), lr=LR, betas=[BETA_1, BETA_2])

    # Helper Function
    def generate_latent(batch_size, latent_dim):
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

    def scale(tensor, mini=-1, maxi=1):
        return tensor * (maxi - mini) + mini

    def scale_back(tensor, mini=-1, maxi=1):
        return (tensor-mini)/(maxi-mini)

    # Generate a fixed latetnt vector. This will be use
    # in monitoring the improvement of generator network
    fixed_z = generate_latent(20, LATENT_DIM)

    # Train Proper
    losses = {"D": [], "G": []}
    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))
        epoch_losses = {"D": [], "G": []}

        start_time = time.time()

        for batch_num, (real_images, _) in enumerate(train_loader):
            # Get the batch size
            batch_size = real_images.shape[0]
            
            # Put the data to the appropriate device
            real_images = real_images.to(device)
            real_images = scale(real_images)

            # Discriminator Real Loss
            d_optim.zero_grad()
            d_real_out = d(real_images)
            d_real_loss = real_loss(d_real_out)

            # Discriminator Fake Loss
            z = generate_latent(batch_size, LATENT_DIM)
            fake_images = g(z)
            d_fake_out = d(fake_images)
            d_fake_loss = fake_loss(d_fake_out)

            # Total Discriminator Loss, Backprop and Gradient Descent
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Generator Loss
            g_optim.zero_grad()
            z = generate_latent(batch_size, LATENT_DIM)
            g_images = g(z)
            d_g_out = d(g_images)
            g_loss = real_loss(d_g_out)

            # Generator Backprop and Gradient Descent
            g_loss.backward()
            g_optim.step()

            # Record Epoch Losses
            epoch_losses["D"].append(d_loss.item())
            epoch_losses["G"].append(g_loss.item())

        # Record Mean Epoch Losses
        losses["D"].append(np.mean(epoch_losses["D"]))
        losses["G"].append(np.mean(epoch_losses["G"]))
        print("D loss: {} G loss: {}".format(d_loss.item(), g_loss.item()))

        # Save Models
        torch.save(g.cpu().state_dict(), MODEL_SAVE_FOLDER + D_NORM_LAYER + str(epoch) + "_G.pth")
        torch.save(d.cpu().state_dict(), MODEL_SAVE_FOLDER + D_NORM_LAYER + str(epoch) + "_D.pth")
        g.to(device)
        d.to(device)

        # Save Images
        g.eval()
        with torch.no_grad():
            sample_fake = g(fixed_z)
            concat_tensor = torchvision.utils.make_grid(sample_fake)
            concat_tensor = scale_back(concat_tensor)
            sample_images = utils.ttoi(concat_tensor.clone().detach())
            
            utils.show(sample_images)
            utils.saveimg(sample_images, SAVE_FOLDER + D_NORM_LAYER + str(epoch) + ".png")
        g.train();

        print("Time Elapsed: {}".format(time.time() - start_time))

    # Plot Network Losses
    def plott(losses):
        fig = plt.figure(figsize=(20,20))
        plt.plot(losses["D"], label="Discriminator")
        plt.plot(losses["G"], label="Generator")
        plt.legend(prop={'size': 20})
        plt.savefig("lossplot.png")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.show()
        plt.close()
    plott(losses)



main()