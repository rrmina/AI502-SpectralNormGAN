import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_layer="SN"):
        super(Discriminator, self).__init__()

        if (norm_layer=="SN"):
            self.discriminator = DiscriminatorSN(conv_dim, norm_layer)
        elif (norm_layer=="BN"):
            self.discriminator = DiscriminatorBaseline(conv_dim, norm_layer)

    def forward(self, x):
        return self.discriminator(x)

class DiscriminatorBaseline(nn.Module):
    def __init__(self, conv_dim, norm_layer):
        super(DiscriminatorBaseline, self).__init__()
        c = conv_dim
        norm = norm_layer
        # Network
        self.network = nn.Sequential(
            ConvLayer(  3,   c, kernel_size=4, stride=2, padding=1, norm_layer=None),   # Input (3, 32, 32)
            ConvLayer(  c, c*2, kernel_size=4, stride=2, padding=1, norm_layer=norm),   # Input: (nc, 16, 16)
            ConvLayer(c*2, c*4, kernel_size=4, stride=2, padding=1, norm_layer=norm),   # Input: (nc*2, 8, 8)
            ConvLayer(c*4, c*8, kernel_size=4, stride=2, padding=1, norm_layer=norm),   # Input: (nc*4, 4, 4)
            ReshapeLayer(shape=(-1, c*8*2*2)),                                          # Input: (nc*8, 2, 2)
            nn.Linear(c*8*2*2, 1)                                                       # Input: (batch_size, nc*8*2*2)
            # Output: (batch_size, 1)
        )

    def forward(self, x):
        return self.network(x)

class DiscriminatorSN(nn.Module):
    def __init__(self, conv_dim, norm_layer):
        super(DiscriminatorSN, self).__init__()
        c = conv_dim

        # Network
        self.network = nn.Sequential(
            ConvLayer(  3,   c, kernel_size=3, stride=1, padding=1, norm_layer="SN"),   # Input (3, 32, 32)
            ConvLayer(  c,   c, kernel_size=4, stride=2, padding=1, norm_layer="SN"),   # SAME PADDING!
            ConvLayer(  c, c*2, kernel_size=3, stride=1, padding=1, norm_layer="SN"),   # Input: (nc, 16, 16)
            ConvLayer(c*2, c*2, kernel_size=4, stride=2, padding=1, norm_layer="SN"),   # SAME PADDING!
            ConvLayer(c*2, c*4, kernel_size=3, stride=1, padding=1, norm_layer="SN"),   # Input: (nc*2, 8, 8)
            ConvLayer(c*4, c*4, kernel_size=4, stride=2, padding=1, norm_layer="SN"),   # SAME PADDING!
            ConvLayer(c*4, c*8, kernel_size=4, stride=2, padding=1, norm_layer="SN"),   # Input: (nc*4, 4, 4)
            ReshapeLayer(shape=(-1, c*8*2*2)),                                          # Input: (nc*8, 2, 2)
            nn.utils.spectral_norm(nn.Linear(c*8*2*2, 1)),                              # Input: (batch_size, nc*8*2*2)
            # Output: (batch_size, 1)
        )

    def forward(self, x):
        return self.network(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, conv_dim):
        super(Generator, self).__init__()
        c = conv_dim

        #  Network
        self.network = nn.Sequential(
            nn.Linear(latent_dim, c*8*2*2),
            ReshapeLayer(shape=(-1, c*8, 2, 2)),
            DeconvLayer(c*8, c*4),
            DeconvLayer(c*4, c*2),
            DeconvLayer(c*2, c),
            DeconvLayer(c,3, bn=False, relu=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=None, relu='leaky'):
        super(ConvLayer, self).__init__()

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # Normalization
        self.norm_layer = norm_layer
        if (norm_layer == "BN"):
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif (norm_layer == "SN"):
            self.conv_layer = nn.utils.spectral_norm(self.conv_layer)

        # Rectified Linear Layer
        self.relu_layer = relu
        if (relu == "relu"):
            self.relu_layer = nn.ReLU()
        elif (relu == "leaky"):
            self.relu_layer = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_layer(x)
        if (self.norm_layer == "BN"):
            x = self.norm_layer(x)
        if (self.relu_layer != None):
            x = self.relu_layer(x)
        return x

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bn=True, relu=True):
        super(DeconvLayer, self).__init__()

        # Transposed Convolution Layer
        self.deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=~bn)

        # Normalization Layer
        self.bn = bn
        if (bn):
            self.norm_layer = nn.BatchNorm2d(out_channels)

        # Activation Layer
        self.relu_layer = relu
        if (relu):
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv_layer(x)
        if (self.bn):
            x = self.norm_layer(x)
        if (self.relu_layer):
            x = self.relu(x)
        return x

def weights_init_xavier_normal(m):
    XAVIER_GAIN = 0.2
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.xavier_normal_(m.weight, gain=XAVIER_GAIN)