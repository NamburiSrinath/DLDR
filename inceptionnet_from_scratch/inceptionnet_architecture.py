import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    Use this conv block to construct the Inception module and build the entire network using that inception module
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs) #kernel size can be (1,1), (3,3) etc;
        # This is not in original paper, but it improves the accuracy, so can just implement it!
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Just pass it through each layer
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        """
        out_1x1 -> The leftmost 1x1 conv layer in inception block (Eg: 64 in inception 3a)
        reduction_3x3 -> the block before 3x3 conv block, basically the 1x1 convolutions output channels which we aim to reduce
        Eg: 96 in inception 3a (see 3x3 reduce heading)
        reduction_5x5 -> The block before 5x5 conv block, basically the 1x1 convolutions output channels, which aims to reduce 
        the channel dimension ()
        Eg: 16 in inception 3a (see 5x5 reduce heading in table 1, page 6)
        out_1x1pool -> Pool projection in table 1 (32 in inception 3a)
        """
        super().__init__()
        # Branch 1 (only 1 conv layer)
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        # Branch 2, has 2 conv blocks
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        # Branch 3 has a 1x1 and 5x5 convolutions
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        # One maxpool and one 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    
    def forward(self, x):
        # We just need to concatenate all the filters (in channel dimension)
        # num_images x filters x 28x28 --> So, concatenation needs to be done in filters dimension
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )

class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super().__init__()

        # 2 layers before the actual inception block starts
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception module
        self.inception3a = inception_block(in_channels=192, out_1x1=64, red_3x3=96, out_3x3=128,
                                            red_5x5=16, out_5x5=32, out_1x1pool=32)

        # The in_channels is same as the output channels from the previous block!!!
        self.inception3b = inception_block(in_channels=256, out_1x1=128, red_3x3=128, out_3x3=192,
                                            red_5x5=32, out_5x5=96, out_1x1pool=64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = inception_block(in_channels=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64)
        self.inception4b = inception_block(in_channels=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64)
        self.inception4c = inception_block(in_channels=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64)
        self.inception4d = inception_block(in_channels=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64)
        self.inception4e = inception_block(in_channels=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inception_block(in_channels=832, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
        self.inception5b = inception_block(in_channels=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        # Pretty straightforward, just pass it through all blocks
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        # Flatten it so that we can do dropout and FC layer
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def test():
    inp = torch.randn(10, 3, 224, 224)
    model = GoogLeNet(in_channels=3)
    print(model(inp).shape)

test()











