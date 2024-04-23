import torch 
import torch.nn as nn
from deep_learning_technique.channel_attention_layer import ChannelAttention 

class Generator(nn.Module):
    def __init__(self, isize, nc, nz, ndf, n_extra_layers=0): # nc=input_channel, nz=output_channel, ndf=number of features
        super(Generator, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # blue part of the network
        # reduce the size of the input image 
        self.e1 = nn.Sequential(
            # reduce the size of the input image by half
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # input_channel, output_channel, kernel_size, stride, padding
            nn.BatchNorm2d(ndf),  #improve the speed, performance, and stability of neural networks, it has a slight regularization effect, similar to dropout, which can help to prevent overfitting. 
            nn.ReLU(True),
            )
        
        # extra layers to increase the number of features(depth) of the network
        self.e_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            # keep the size of the input image the same
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = nn.Sequential(
            # reduce the size of the input image by half
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True), # here we use LeakyReLU instead of ReLU to avoid the dying ReLU problem(when the input is negative, the output is zero, which causes the gradient to be zero and the network cannot be trained)
                                             # 0.2 is the slope of the negative part of the function and inplace=True means that the input is modified directly, without allocating additional memory 
            )
        self.e3 = nn.Sequential(
            # reduce the size of the input image by half
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e4 = nn.Sequential(
            # keep the size of the input image the same
            nn.Conv2d(ndf*4, nz, 3, 1, 1, bias=False),
            )
        
        # orange part of the network
        # increase the size of the input 
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(nz, ndf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4*2, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),
            )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.d_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(ndf*2, ndf),
                            nn.Conv2d(ndf*2, ndf, 3, 1, 1, bias=False))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(ndf, ndf),
                            nn.BatchNorm2d(ndf))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(ndf, ndf),
                            nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2, 1, 4, 2, 1, bias=False),
#             nn.LeakyReLU(),
            nn.Sigmoid(),
#             nn.ReLU(),
            )
        #attention module

        self.at1 = ChannelAttention(1)
        self.at2 = ChannelAttention(1)
        self.at3 = ChannelAttention(1)
        self.at4 = ChannelAttention(1)
        
    def forward(self,x):
        # blue part of the network
        e1 = self.e1(x)
        e_el = self.e_extra_layers(e1)
        e2 = self.e2(e_el)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        # orange part of the network
        d4 = self.d4(e4)
        # attention module
        d4 = self.at4(d4)
        # concatenate the output of the blue part and the orange part
        c34 = torch.cat((e3,d4),1)
        # second orange part of the network
        d3 = self.d3(c34)
        # attention module
        d3 = self.at3(d3)
        # concatenate the output of the second orange part and the second orange part
        c23 = torch.cat((e2,d3),1)

        d2 = self.d2(c23)
        d2 = self.at2(d2)
        
        cel2 = torch.cat((e_el,d2),1)
        d_el = self.d_extra_layers(cel2)
        e_el = self.at1(d_el)

        c11 = torch.cat((e1,d_el),1)
        d1 = self.d1(c11)
        
        return d1
    
# print the model summary
from torchsummary import summary
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Generator(256, 3*2,100, 64, 3)
  y = model(torch.randn(2,6,256,256))    
  print(y.shape)
  model = model.to(device)
  summary(model, (6,256,256))