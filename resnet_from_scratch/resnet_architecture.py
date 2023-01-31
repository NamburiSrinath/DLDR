import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, input_channels, output_channels, identity_downsample=None, stride=1):
        """
        Identity downsample -> 
        """
        super().__init__()
        # Number of channels increases by 4 times by the end of the block 
        # (see paper conv_2x, conv_3x start and end channel numbers)
        self.expansion = 4
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(output_channels)

        # The input dimension of 2nd conv will be the output of the 1st block. And can't reduce in height and width, 
        # so let's pad    
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)


        # The output channels dimension will be 4 times the input channel dimension (observe the table from paper)
        self.conv3 = nn.Conv2d(output_channels, output_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(output_channels*self.expansion)

        self.relu = nn.ReLU()
        # This is basically the resnet skip connection
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x.clone()

        # Pass x through one block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # Incase we want to change the identity before we add it to signal
        # Used when going from one conv block to other conv block (conv2_x to conv3_x when increasing dimension)
        # Increase the channel dimension (either via identity shortcut, where we just add) or by 
        # projection shortcut where we multiply with a weight matrix before adding (Equation 2 in paper)
        if self.identity_downsample is not None:
            # This is projection shortcut (Option B in paper, page 6)
            identity = self.identity_downsample(identity)

        # Skip connection
        x = x + identity
        x = self.relu(x)
        return x
    

class Resnet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        # layers -> [3, 4, 6, 3]. Number of times each block gets repeated
        super().__init__()
        # Initial number of channels in the first conv_1
        self.in_channels = 64
        # See the conv_1 from Table 1 in paper (this entire thing will be in conv1 of table 1)
        # The output shape is half of input shape, that indirectly implies that we have a padding of 3
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        # Once we pass the values through a conv
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


    # num_residual_blocks -> number of convs inside a residual block
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # We need to change the identity whenever we go from one ResNet block to other ResNet block
        # In paper we have 2 types -> Identity mapping and Projection mapping (see --- and solid lines)
        # Basically the skip connection from conv2x is different from skip connection from conv3x

        # This is the projection mapping we do once in every layer
        # Not sure if this condition is even needed or not because identity downsample is getting created everytime!!
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels*4, kernel_size=1,
                    stride=stride 
                ),
                nn.BatchNorm2d(out_channels*4)
            )
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        # Till above we implemented conv2x!!

        # We are doing this step because the no of channels are 256, but we want them to convert to 64
        # So, input will be 256, output will be 64
        self.in_channels = out_channels * 4 #which will be 256

        # -1 because we already added conv2x
        for i in range(num_residual_blocks - 1):
            # Input 256, output 64 (see near the joininig of 2 blocks)
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)


def ResNet50(image_channels=3, num_classes=1000):
    return Resnet(block, [3, 4, 6, 3], image_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()

