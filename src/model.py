import torch
import torch.nn as nn

import torchvision


class SUM_2D(nn.Module):

    def __init__(self):
        super(SUM_2D,self).__init__()
        self.backbone = torchvision.models.resnet18(num_classes=1)
        #self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), bias=False)

    def forward(self, inputs):

        x = self.backbone(inputs)

        return x.cpu()


class R2plus1D_18(nn.Module):

    def __init__(self):
        super(R2plus1D_18,self).__init__()

        self.backbone = torchvision.models.video.r3d_18()
        self.backbone.stem = nn.Sequential(
            nn.Conv3d(3,64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,0,0)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.backbone.fc = nn.Linear(512,1)
        
    def forward(self, inputs):

        x = self.backbone(inputs)
        
        return x.cpu()

class CNN_3D(nn.Module):

    def __init__(self):
        super(CNN_3D,self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv3d(3,10,(100,5,5),(1,2,2),0),
            nn.ReLU(),
            nn.BatchNorm3d(10),
            nn.MaxPool3d((1,5,5))
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(3,10,(20,5,5),(5,2,2),0),
            nn.ReLU(),
            nn.BatchNorm3d(10),
            nn.MaxPool3d((2,5,5))
        )

        self.c3 = nn.Sequential(
            nn.Conv3d(3,10,(5,5,5),(2,2,2),0),
            nn.ReLU(),
            nn.BatchNorm3d(10),
            nn.MaxPool3d((3,5,5))
        )
        
        self.l1 = nn.Linear(20250,5000)
        self.l2 = nn.Linear(5000,1000)
        self.l3 = nn.Linear(1000,100)
        self.fc = nn.Linear(100,1)

    def forward(self, inputs):

        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)

        x = torch.cat([x1,x2,x3],dim=2)

        x = x.reshape(x.shape[0],-1)
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.fc(x)
        
        return x.cpu()





class CNN_LSTM(nn.Module):
    
    def __init__(self):
        super(CNN_LSTM,self).__init__()

        self.backbone = torchvision.models.squeezenet1_0(pretrained=True)
        self.backbone.features[0] = nn.Conv2d(1,96,(7,7),stride=(2,2))

        input_size = 1000
        hidden_size = 1000
        n_layers = 1
        seq_len = 100
        batch_size = 5

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        

        #inp = torch.randn(batch_size, seq_len, input_dim)
        hidden_state = torch.zeros(n_layers, batch_size, hidden_size).cuda()
        cell_state = torch.zeros(n_layers, batch_size, hidden_size).cuda()
        self.hidden = (hidden_state, cell_state)


        self.fc = nn.Linear(100000,1)


    def forward(self, inputs):

        x = []

        for inp in inputs:
            frames = self.backbone(inp.unsqueeze(1))
            x.append(frames)

        x = torch.stack(x,dim=0)

        out, self.hidden = self.lstm(x,self.hidden)


        x = out.reshape(5,-1)

        x = self.fc(x)
        
        return x.cpu()
