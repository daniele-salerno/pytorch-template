import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# TODO: move to custom layers
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        self.depthwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3,3), groups=in_channels)
        self.pointwise_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise_layer(x)
        x = self.pointwise_layer(x)
        return x


class LModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DepthSepWiseNeuralNetwork(BaseModel):
    """debug with x.shape and check the size of tensors

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self,channels=3, num_classes=10) -> None:
        """_summary_

        Args:
            channels (int, optional): _description_. Defaults to 3.
            num_classes (int, optional): _description_. Defaults to 10.
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3,3))
        self.relu = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv2d(in_channels=32, out_channels=64)
        # relu
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = DepthwiseSeparableConv2d(in_channels=64, out_channels=64, )
        # relu
        self.conv4 = DepthwiseSeparableConv2d(in_channels=64, out_channels=8,)
        # relu
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1352, out_features=256)
        # relu
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)

        
        
    def forward(self, x):
        x = self.conv1(x) # out: (BS, 32, 30, 30)
        x = self.relu(x)
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.pool1(x) 
        x = self.dropout1(x) # ([BS, 64, 14, 14])
        
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.conv4(x) # ([BS, 8, 5, 5])
        x = self.relu(x)
        x = self.pool2(x) 
        x = self.dropout2(x) 
        
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x) 
        
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class ConvolutionalNeuralNetwork(BaseModel):
    """debug with x.shape and check the size of tensors

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self,channels=3, num_classes=10) -> None:
        """_summary_

        Args:
            channels (int, optional): _description_. Defaults to 3.
            num_classes (int, optional): _description_. Defaults to 10.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3,3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        # relu
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3))
        #relu
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3,3))
        # relu
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8*13*13, out_features=256)
        # relu
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim = 1)

        
        
    def forward(self, x):
        x = self.conv1(x) # out: (BS, 32, 30, 30)
        x = self.relu(x)
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.pool1(x) 
        x = self.dropout1(x) # ([BS, 64, 14, 14])
        
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.conv4(x) # ([BS, 8, 5, 5])
        x = self.relu(x)
        x = self.pool2(x) 
        x = self.dropout2(x) 
        
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x) 
        
        x = self.fc2(x)
        x = self.softmax(x)
        return x
