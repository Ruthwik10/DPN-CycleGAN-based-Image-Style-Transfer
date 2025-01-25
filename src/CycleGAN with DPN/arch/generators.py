import functools
import torch
from torch import nn
from .ops import conv_norm_relu, dconv_norm_relu, ResidualBlock, get_norm_layer, init_network


def conv_norm_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        norm_layer(out_channels),
        nn.ReLU(True)
    )

def dconv_norm_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, norm_layer=nn.BatchNorm2d, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        norm_layer(out_channels),
        nn.ReLU(True)
    )

# Shivaji Panam made changes to the parameters of designed architecutre
class DPNBlock(nn.Module):           # Ruthwik created DPN Class
    def __init__(self, in_channels, out_channels):
        super(DPNBlock, self).__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2)
        )
        
        self.dense_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        residual = self.residual_path(x)
        dense = self.dense_path(x)
        
        # Concatenate both paths (dual path)
        return torch.cat([residual + x[:, :residual.size(1)], dense], dim=1)

# Ruthwik Adapala designed model architecutre based on research paper model diagram.
class DPNGenerator(nn.Module):
    def __init__(self , input_nc=3 , output_nc=3 , ngf=64 , norm_layer=nn.BatchNorm2d):
        super(DPNGenerator , self).__init__()

        # Initial convolutional layers
        self.conv1 = conv_norm_relu(input_nc , ngf * 1 , kernel_size=7 , padding=3)
        self.conv2 = conv_norm_relu(ngf * 1 , ngf * 2 , kernel_size=3 , stride=2 , padding=1)
        self.conv3 = conv_norm_relu(ngf * 2 , ngf * 4 , kernel_size=3 , stride=2 , padding=1)

        # Add multiple DPN blocks (5 as per the image)   # Shivaji Added 5 blocks of DPN by following research paper.
        self.dpn_blocks = nn.Sequential(*[DPNBlock(ngf *4 , ngf *4) for _ in range(5)])

        # Upsampling layers (deconvolution)
        self.deconv1 = dconv_norm_relu(ngf *4 , ngf *2 , kernel_size=3 , stride=2 , padding=1)
        self.deconv2 = dconv_norm_relu(ngf *2 , ngf *1 , kernel_size=3 , stride=2)

        self.pad = nn.Identity()   # Ruthwik Adapala added this to resolve conflict between input image shape and output image shape
        self.final_conv = nn.Conv2d(ngf *1 , output_nc , kernel_size=7 , padding=3)  
        self.tanh = nn.Tanh()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)

        for i in range(5):
            x=self.dpn_blocks[i](x)

        x=self.deconv1(x)
        x=self.deconv2(x)
        
        x=self.pad(x)  
        x=self.final_conv(x)
        x=self.tanh(x)

        return x

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                                innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [nn.ReLU(True), upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [nn.LeakyReLU(0.2, True), downconv, norm_layer(inner_nc)]
            up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.unet_model = unet_block

    def forward(self, input):
        return self.unet_model(input)



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, num_blocks=6):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        res_model = [nn.ReflectionPad2d(3),
                    conv_norm_relu(input_nc, ngf * 1, 7, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 1, ngf * 2, 3, 2, 1, norm_layer=norm_layer, bias=use_bias),
                    conv_norm_relu(ngf * 2, ngf * 4, 3, 2, 1, norm_layer=norm_layer, bias=use_bias)]

        for i in range(num_blocks):
            res_model += [ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)]

        res_model += [dconv_norm_relu(ngf * 4, ngf * 2, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      dconv_norm_relu(ngf * 2, ngf * 1, 3, 2, 1, 1, norm_layer=norm_layer, bias=use_bias),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(ngf, output_nc, 7),
                      nn.Tanh()]
        self.res_model = nn.Sequential(*res_model)

    def forward(self, x):
        return self.res_model(x)

# Define a convolutional block with normalization and ReLU activation


def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, gpu_ids=[0]):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'dpn_gen':      # Ruthwik Adapala
        print('Selected DPN Generator')
        gen_net = DPNGenerator(input_nc , output_nc , ngf , norm_layer)
    elif netG == 'resnet_6blocks':
        gen_net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, num_blocks=6)
    elif netG == 'unet_128':
        gen_net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        gen_net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net, gpu_ids)