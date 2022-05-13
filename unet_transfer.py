from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
import time
# writer = SummaryWriter('res34unet')


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels))


    def forward(self, x):
        return self.block(x)



class Bottleneck(nn.Module):
  def __init__(self,in_places,places, stride=1, downsampling=False):
    super(Bottleneck,self).__init__()
    self.downsampling = downsampling

    self.bottleneck = nn.Sequential(
      nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=3,stride=stride, padding=1, bias=False),
      nn.BatchNorm2d(places),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(places),
    )

    if self.downsampling:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(places)
      )
    self.relu = nn.ReLU(inplace=True)


  def forward(self, x):
    residual = x
    out = self.bottleneck(x)

    if self.downsampling:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out


class VGG16Unet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes: output channel
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super(VGG16Unet,self).__init__()

        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        #print(torchvision.models.vgg16(pretrained=pretrained))
        # self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(2, 2))

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        # self, in_channels, middle_channels, out_channels, is_deconv = True
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            #x_out = F.sigmoid(x_out)

        return x_out


class Res34Unet(nn.Module):

    def __init__(self,  in_channels, num_classes, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super(Res34Unet,self).__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        bottom_channel_nr = 512
        blocks = [3, 4, 6, 3]

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                   nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.conv3 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
        self.conv4 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
        self.conv5 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))  # stride为输入值1，2，2，2
        for i in range(1, block):
            layers.append(Bottleneck(places, places))  # stride为默认值1
        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)


        dec5 = self.dec5(torch.cat([center, conv5], 1)) #channel维度拼接
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


if __name__ == '__main__':
    start = time.time()
    # net = VGG16Unet(in_channels=1, num_classes = 2, num_filters = 32, pretrained = False, is_deconv = False)
    net = Res34Unet(in_channels = 1, num_classes = 2, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False)
    img = torch.randn(1, 1, 512, 512)  # b c h w
    # print(net)

    # writer.add_graph(net, (img,))

    output = net(img)
    print(output.shape)
    # writer.close()
    end = time.time()
    print("运行时间：{}s".format(end-start))