"""
Reference: https://www.youtube.com/watch?v=1ZbLA7ofasY

When you import Module in Pytorch like
class AAA(nn.Module):
    super().__init__() --> This line inherits the variables that are defined in Module class in Pytorch
    And one variable defined there is _forward_hooks.

Also, why we shouldn't need to call forward()!!
It gets called inside a dunder method __call__() which in addition to self.forward() does a whole lot of things

Approach 1 -> Create a SummaryWriter instance and store logs inside it as a histogram -> Not optimal
Approach 2 -> Use forward hooks
"""
# Approach 1 -> Commented whole
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

# Stuff used for logging and visualizing
import pathlib
from torch.utils.tensorboard import SummaryWriter

class Network(Module):
    def __init__(self): # This is the constructor of the subclass
        super().__init__() 
        # We are calling the constructor of the parent i.e all thosse self.parameters in parent will be inherited

        # Define the network layers
        self.fc1 = Linear(10, 20)
        self.fc2 = Linear(20, 30)
        self.fc3 = Linear(30, 2)

        # This creates a folder and a place where the logs get stored. An instance is created at self.writer
        log_dir = pathlib.Path.cwd() / "tensorboard_logs"
        self.writer = SummaryWriter(log_dir)
    
    def forward(self, x):
        x = self.fc1(x)
        self.writer.add_histogram("1", x)
        x = self.fc2(x)
        self.writer.add_histogram("2", x)
        x = self.fc3(x)
        self.writer.add_histogram("3", x)
        x = F.relu(x)

        return x

if __name__ == "__main__":
    x = torch.randn(3, 10)
    net = Network()
    y = net(x)
    print(y)
"""

# Approach 2 -> Use forward hooks
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

# Stuff used for logging and visualizing
import pathlib
from torch.utils.tensorboard import SummaryWriter

class Network(Module):
    def __init__(self): # This is the constructor of the subclass
        super().__init__() 
        # We are calling the constructor of the parent i.e all thosse self.parameters in parent will be inherited

        # Define the network layers
        self.fc1 = Linear(10, 20)
        self.fc2 = Linear(20, 30)
        self.fc3 = Linear(30, 2)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)

        return x


def activation_hook(inst, inp, out):
    """
    instance -> The layer we want to attach our hook on
    inp -> Input to the forward method
    out -> Output from the forward method
    """
    # repr stands for representation/name of the layer
    writer.add_histogram(repr(inst), out)


if __name__ == "__main__":
    # This creates a folder and a place where the logs get stored. An instance is created at self.writer
    log_dir = pathlib.Path.cwd() / "tensorboard_logs_forward_hooks"
    writer = SummaryWriter(log_dir)
    x = torch.randn(3, 10)

    net = Network()

    # Add a hook to the 1st FC layer, so we can see the outputf from this layer
    # register_forward_hook will take care that the function we mentioned will be run in forward hook
    handle1 = net.fc1.register_forward_hook(activation_hook)
    handle2 = net.fc2.register_forward_hook(activation_hook)
    handle3 = net.fc3.register_forward_hook(activation_hook)

    # Now as we run via 3 layers, 3 hooks gets executed and we can store all activations
    y = net(x)

    # Suppose we want to remove one hook from one particular layer
    handle1.remove() # This will unregister the forward hook

    # This will only store the activations from 2,3rd layers
    y = net(x)
