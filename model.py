

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        #IMPORT MODEL
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)


        #Transfer Learning
        for p in self.resnet.parameters():
            p.requires_grad = False



        #Unfreeze the parameters of the last few layers for fine-tuning
        for param in resnet.layer4.parameters():
            param.requires_grad=True
        '''
        for param in resnet.layer3.parameters():
            param.requires_grad=True

        for param in resnet.layer2.parameters():
            param.requires_grad=True
        '''



        #Decoder
        self.conv1 = nn.Sequential(
                     nn.ConvTranspose2d(2048,512,kernel_size=4,stride=2,padding=1),
                     nn.BatchNorm2d(512),
                     nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        self.decoder = nn.Sequential(
                        self.conv1,
                        self.conv2,
                        self.conv3,
                        self.conv4,
                        nn.ConvTranspose2d(64,1,4,2,1)
        )

        self.model = nn.Sequential(self.resnet,
                                   self.decoder)


    def forward(self,x):

        #MODEL
        depth = self.model(x)
        #print('Output resnet50: ',depth.shape)
        return depth


