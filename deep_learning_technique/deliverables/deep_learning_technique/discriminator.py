import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0):
        super(Discriminator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.e1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # nc is the input channel 1
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.e_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.toplayer = nn.Sequential(
            nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            ) 
        self.avgpool = nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            )
    def forward(self,x):
        x = self.e1(x)
        x = self.e_extra_layers(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.toplayer(x)
        x = self.avgpool(x)
        x = x.view(-1,1).squeeze(1)
        return x
    
# print the model summary
from torchsummary import summary
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Discriminator(256,1,100, 64, 3)
  y = model(torch.randn(2,1,256,256))    
  print(y.shape)
  model = model.to(device)
  summary(model, (1,256,256))