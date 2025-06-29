
import torch
from torch import nn
class AlexNet(nn.Module):
    

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,),
                                    nn.ReLU(),
                                    nn.LocalResponseNorm(size=5,alpha =0.001,beta=0.75,k=1),
                                    nn.MaxPool2d(3,2)
        )

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5),
                                    nn.ReLU(),
                                    nn.LocalResponseNorm(5,0.001,0.75,1),
                                    nn.MaxPool2d(3,2))

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels= 256,out_channels=384,kernel_size=3),
                                    nn.ReLU()
        )

        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3),
                                    nn.ReLU())

        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3),
                                    nn.ReLU(),
                                    nn.Flatten())
            

        self.fc = nn.Sequential(nn.Dropout(),
                                nn.Linear(in_features=4096,out_features=4096),
                                nn.ReLU(),
                                nn.Linear(in_features=4096,out_features=4096),
                                nn.ReLU(),
                                nn.Linear(in_features=4096,out_features=10)
                                )
  
    def forward(self,x):

        return (self.fc(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))))
    


