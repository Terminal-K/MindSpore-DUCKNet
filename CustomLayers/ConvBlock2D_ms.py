# from keras.layers import BatchNormalizationV2, add
# from keras.layers import Conv2D

from mindspore.common.initializer import HeUniform
import mindspore.nn as nn

class Duckv2Conv2dBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, size=6):
        super(Duckv2Conv2dBlock, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels)
        self.wide_conv = WidescopeConv2dBlock(in_channels=in_channels, out_channels=out_channels)
        self.mid_conv = MidscopeConv2dBlock(in_channels=in_channels, out_channels=out_channels)
        self.res_conv1 = ResnetConv2dBlock(in_channels=in_channels, out_channels=out_channels)
        self.res_conv2 = nn.SequentialCell([ResnetConv2dBlock(in_channels=in_channels, out_channels=out_channels),
                                            ResnetConv2dBlock(in_channels=out_channels, out_channels=out_channels)])
        self.res_conv3 = nn.SequentialCell([ResnetConv2dBlock(in_channels=in_channels, out_channels=out_channels),
                                            ResnetConv2dBlock(in_channels=out_channels, out_channels=out_channels),
                                            ResnetConv2dBlock(in_channels=out_channels, out_channels=out_channels)])
        self.separated_conv = SeparatedConv2dBlock(in_channels=in_channels, out_channels=out_channels, size=size)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def construct(self, x):
        x = self.bn0(x)
        x1 = self.wide_conv(x)
        x2 = self.mid_conv(x)
        x3 = self.res_conv1(x)
        x4 = self.res_conv2(x)
        x5 = self.res_conv3(x)
        x6 = self.separated_conv(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn1(x)

        return x

class SeparatedConv2dBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, size=3):
        super(SeparatedConv2dBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, size), weight_init=HeUniform())
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(size, 1), weight_init=HeUniform())
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def construct(self, x):
        x1 = self.bn1(self.act(self.conv1(x)))
        x2 = self.bn2(self.act(self.conv2(x1)))

        return x2

class MidscopeConv2dBlock(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(MidscopeConv2dBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def construct(self, x):
        x1 = self.bn1(self.act(self.conv1(x)))
        x2 = self.bn2(self.act(self.conv2(x1)))

        return x2

class WidescopeConv2dBlock(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(WidescopeConv2dBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=3)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def construct(self, x):
        x1 = self.bn1(self.act(self.conv1(x)))
        x2 = self.bn2(self.act(self.conv2(x1)))
        x3 = self.bn3(self.act(self.conv3(x2)))

        return x3

class ResnetConv2dBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResnetConv2dBlock, self).__init__()

        self.skip_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, weight_init=HeUniform(), dilation=dilation)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, weight_init=HeUniform(), dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def construct(self, x):
        skip = self.skip_conv(x)

        x1 = self.bn1(self.act(self.conv1(x)))
        x2 = self.bn2(self.act(self.conv2(x1)))

        res = self.bn3(x2 + skip)
        return res
