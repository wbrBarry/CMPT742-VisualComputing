import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line

class twoConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(twoConvBlock, self).__init__()
        # Initialize the first convolution, ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Initialize the second convolution, BatchNorm, ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass for first conv, ReLU
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Forward pass for second conv, BatchNorm, ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x

class downStep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downStep, self).__init__()
        # Initialize the down path with a twoConvBlock followed by a max-pooling layer
        self.conv_block = twoConvBlock(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Forward pass through the convolution block then max-pooling
        x = self.conv_block(x)
        x_pooled = self.max_pool(x)
        return x, x_pooled

class upStep(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(upStep, self).__init__()
        # Initialize the up path with a transpose convolution (upsampling)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Initialize the twoConvBlock with the concatenated channels
        self.conv_block = twoConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        # Upsample the input feature map
        x = self.up_conv(x)
        # Concatenate the upsampled feature map with the skip connection feature map
        x = torch.cat((x, skip_connection), dim=1)
        # Pass the concatenated feature maps through the convolution block
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting Path
        self.down1 = downStep(1, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)

        # Bottom layer of the U-Net without pooling
        self.bottom = twoConvBlock(512, 1024)

        # Expansive Path
        self.up4 = upStep(1024, 512, 512)
        self.up3 = upStep(512, 256, 256)
        self.up2 = upStep(256, 128, 128)
        self.up1 = upStep(128, 64, 64)

        # Final output layer to get the segmentation map
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Contracting Path
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)

        # Bottom layer
        x = self.bottom(x)

        # Expansive Path with skip connections
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # Final output
        x = self.sigmoid(self.final_conv(x))
        return x