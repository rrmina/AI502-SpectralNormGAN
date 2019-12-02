import torch
import torch.nn
import torchvision

from models import Discriminator, Generator
import utils

MODEL_PATH = "SNGAN/SN50_G.pth" # "DCGAN_Baseline/BN50_G.pth" # "SNGAN/SN50_G.pth"
NORM_LAYER = "SN"
LATENT_DIM = 128
CONV_DIM = 64
NUM_IMAGES = 100

def generate():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load network
    g = Generator(LATENT_DIM, CONV_DIM)
    g.load_state_dict(torch.load(MODEL_PATH))
    g = g.to(device)

    # Helper Functions
    def scale_back(tensor):
        return (tensor+1) / 2

    def generate_latent_uniform(batch_size, latent_dim, device):
        return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

    z = generate_latent_uniform(NUM_IMAGES, LATENT_DIM, device)

    with torch.no_grad():
        torch.cuda.empty_cache()

        # Generate image tensor | Transform tensor to numpy arrays
        generated_tensor = g(z)
        concat_tensor = torchvision.utils.make_grid(generated_tensor, nrow=10)
        concat_tensor = scale_back(concat_tensor)
        generated_images = utils.ttoi(concat_tensor)

        # Save
        filename = NORM_LAYER + ".png"
        utils.saveimg(generated_images, filename)

generate()