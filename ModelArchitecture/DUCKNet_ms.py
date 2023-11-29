# import tensorflow as tf
# from keras.layers import Conv2D, UpSampling2D
# from keras.layers import add
# from keras.models import Model

import mindspore.nn as nn
from CustomLayers.ConvBlock2D_ms import Duckv2Conv2dBlock, ResnetConv2dBlock

class DuckNet(nn.Cell):
    def __init__(self, img_height, img_width, input_chanels, out_classes, starting_filters=17, interpolation="nearest"):
        super(DuckNet, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.input_chanels = input_chanels
        self.out_classes = out_classes
        self.starting_filters = starting_filters

        self.p1 = nn.Conv2d(in_channels=input_chanels, out_channels=starting_filters * 2, kernel_size=2, stride=2)
        self.p2 = nn.Conv2d(in_channels=starting_filters * 2, out_channels=starting_filters * 4, kernel_size=2, stride=2)
        self.p3 = nn.Conv2d(in_channels=starting_filters * 4, out_channels=starting_filters * 8, kernel_size=2, stride=2)
        self.p4 = nn.Conv2d(in_channels=starting_filters * 8, out_channels=starting_filters * 16, kernel_size=2, stride=2)
        self.p5 = nn.Conv2d(in_channels=starting_filters * 16, out_channels=starting_filters * 32, kernel_size=2, stride=2)

        self.l1i = nn.Conv2d(in_channels=starting_filters, out_channels=starting_filters * 2, kernel_size=2, stride=2)
        self.l2i = nn.Conv2d(in_channels=starting_filters * 2, out_channels=starting_filters * 4, kernel_size=2, stride=2)
        self.l3i = nn.Conv2d(in_channels=starting_filters * 4, out_channels=starting_filters * 8, kernel_size=2, stride=2)
        self.l4i = nn.Conv2d(in_channels=starting_filters * 8, out_channels=starting_filters * 16, kernel_size=2, stride=2)
        self.l5i = nn.Conv2d(in_channels=starting_filters * 16, out_channels=starting_filters * 32, kernel_size=2, stride=2)

        self.t0 = Duckv2Conv2dBlock(in_channels=input_chanels, out_channels=starting_filters)
        self.t1 = Duckv2Conv2dBlock(in_channels=starting_filters * 2, out_channels=starting_filters * 2)
        self.t2 = Duckv2Conv2dBlock(in_channels=starting_filters * 4, out_channels=starting_filters * 4)
        self.t3 = Duckv2Conv2dBlock(in_channels=starting_filters * 8, out_channels=starting_filters * 8)
        self.t4 = Duckv2Conv2dBlock(in_channels=starting_filters * 16, out_channels=starting_filters * 16)
        self.t51 = nn.SequentialCell(
            [ResnetConv2dBlock(in_channels=starting_filters * 32, out_channels=starting_filters * 32),
             ResnetConv2dBlock(in_channels=starting_filters * 32, out_channels=starting_filters * 32)])
        self.t53 = nn.SequentialCell(
            [ResnetConv2dBlock(in_channels=starting_filters * 32, out_channels=starting_filters * 16),
             ResnetConv2dBlock(in_channels=starting_filters * 16, out_channels=starting_filters * 16)])

        self.l5o = nn.Upsample(size=(img_height // 2**4), mode=interpolation)
        self.l4o = nn.Upsample(size=(img_height // 2**3), mode=interpolation)
        self.l3o = nn.Upsample(size=(img_height // 2**2), mode=interpolation)
        self.l2o = nn.Upsample(size=(img_height // 2**1), mode=interpolation)
        self.l1o = nn.Upsample(size=(img_height // 2**0), mode=interpolation)

        self.q4 = Duckv2Conv2dBlock(in_channels=starting_filters * 16, out_channels=starting_filters * 8)
        self.q3 = Duckv2Conv2dBlock(in_channels=starting_filters * 8, out_channels=starting_filters * 4)
        self.q2 = Duckv2Conv2dBlock(in_channels=starting_filters * 4, out_channels=starting_filters * 2)
        self.q1 = Duckv2Conv2dBlock(in_channels=starting_filters * 2, out_channels=starting_filters)

        self.z1 = Duckv2Conv2dBlock(in_channels=starting_filters, out_channels=starting_filters)

        self.output = nn.Conv2d(in_channels=starting_filters, out_channels=out_classes, kernel_size=1)
        self.act = nn.Sigmoid()

    def construct(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(x)
        s1 = self.l1i(t0) + p1

        t1 = self.t1(s1)
        s2 = self.l2i(t1) + p2

        t2 = self.t2(s2)
        s3 = self.l3i(t2) + p3

        t3 = self.t3(s3)
        s4 = self.l4i(t3) + p4

        t4 = self.t4(s4)
        s5 = self.l5i(t4) + p5

        t51 = self.t51(s5)
        t53 = self.t53(t51)

        c4 = self.l5o(t53) + t4
        q4 = self.q4(c4)

        c3 = self.l4o(q4) + t3
        q3 = self.q3(c3)

        c2 = self.l3o(q3) + t2
        q2 = self.q2(c2)

        c1 = self.l2o(q2) + t1
        q1 = self.q1(c1)

        c0 = self.l1o(q1) + t0
        z1 = self.z1(c0)

        output = self.act(self.output(z1))

        return output

if __name__ == "__main__":
    import mindspore as ms
    image_size = 352
    input = ms.ops.rand((1, 3, image_size, image_size))
    print(input.shape)
    model = DuckNet(img_height=image_size, img_width=image_size, input_chanels=3, out_classes=1)
    output = model(input)

    print(output.shape)