import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DecoderBlock, self).__init__()
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)
        
class ContextEncoder(nn.Module):
     def __init__(self, in_channels=3, bottleneck_size=4000):
        super(ContextEncoder, self).__init__()
        
        # input 3 x 128 x 128 x images
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # data shape = 64 x 64 x 64
        self.block1 = EncoderBlock(in_channels=64, out_channels=64)
        
        #data shape = 64 x 32 x 32
        self.block2 = EncoderBlock(in_channels=64, out_channels=128)
        
        #data shape = 128 x 16 x 16
        self.block3 = EncoderBlock(in_channels=128, out_channels=256)
        
        #data shape = 256 x 8 x 8
        self.block4 = EncoderBlock(in_channels=256, out_channels=512)
        
        #data shape = 512 x 4 x 4
        self.bottle_neck = EncoderBlock(in_channels=512, out_channels=bottleneck_size, stride=1, padding=0)
        
        #data shape bottleneck_size = 4000 x 1 x 1
    
     def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)


        return self.bottle_neck(x)

class ContextDecoder(nn.Module):
    def __init__(self, bottleneck_size=4000, out_channels=3):
        super(ContextDecoder, self).__init__()
        
        #input is output from bottleneck 4000 x 1 x 1
        self.bottle_neck = DecoderBlock(in_channels=bottleneck_size, out_channels=512, stride=1, padding=0)
        
        # data shape = 512 x 4 x 4
        self.block1 = DecoderBlock(in_channels=512, out_channels=256)
        
        # data shape = 256 x 8 x 8
        self.block2 = DecoderBlock(in_channels=256, out_channels=128)
        
        # data shape = 128 x 16 x 16
        self.block3 = DecoderBlock(in_channels=128, out_channels=64)
        
        # data shape = 64 x 32 x 32
        self.block4 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        # data shape = 3 x 64 x 64
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.bottle_neck(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return self.tanh(x)

class ContextInpainter(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bottleneck_size=4000):
        super(ContextInpainter, self).__init__()
        
        # input: batch_size x 3 x 128 x 128
        self.context_encoder = ContextEncoder(in_channels=in_channels, bottleneck_size=bottleneck_size)
        
        # data shape = batch_size x bottleneck_size x 1 x 1
        self.context_decoder = ContextDecoder(bottleneck_size=bottleneck_size, out_channels=out_channels)
        
        # output: batch_size x 3 x 64 x 64
        
    def forward(self, x):
        x = self.context_encoder(x)
        return self.context_decoder(x)
   
        
class AdversarialDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(AdversarialDiscriminator, self).__init__()
        # input: batch_size x 3 x 64 x 64
        
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        #data_shape: batch_size x 64 x 32 x 32
        self.block1 = EncoderBlock(in_channels=64, out_channels=128)
        
        #data_shape: batch_size x 128 x 16 x 16
        self.block2 = EncoderBlock(in_channels=128, out_channels=256)
        
        #data_shape: batch_size x 256 x 8 x 8
        self.block3 = EncoderBlock(in_channels=256, out_channels=512)
        
        #data_shape: batch_size x 512 x 4 x 4
        self.decision = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        
        # data_shape: batch_size x 1 x 1 x 1
        self.sig = nn.Sigmoid()
        
        #output: real or fake
        
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        return self.sig(self.decision(x))