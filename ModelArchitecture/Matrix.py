import numpy as np
import mindspore.train as train

class PixelAccuracy(train.Metric):
    def __init__(self, batch_size=4, num_class=1, img_size=352):
        super(PixelAccuracy, self).__init__()
        self.batch_size = batch_size
        self.num_class = num_class
        self.img_size = img_size

    def clear(self):
        self.acc = []

    def update(self, *inputs):
        y_pred = np.where(inputs[0].asnumpy() > 0.5, 1., 0.).squeeze(1).astype(np.float32)
        y = inputs[1].asnumpy().reshape(-1, self.img_size, self.img_size)
        TP = ((y_pred == 1) & (y == 1)).sum().item()
        TN = ((y_pred == 0) & (y == 0)).sum().item()
        FP = ((y_pred == 1) & (y == 0)).sum().item()
        FN = ((y_pred == 0) & (y == 1)).sum().item()
        self.acc.append((TP + TN) / (TP + TN + FP + FN))

    def eval(self):
        pixel_acc = sum(self.acc) / len(self.acc)
        return pixel_acc

class IntersectionOverUnion(train.Metric):
    def __init__(self, batch_size=4, num_class=1, img_size=352):
        super(IntersectionOverUnion, self).__init__()
        self.batch_size = batch_size
        self.num_class = num_class
        self.img_size = img_size

    def clear(self):
        self.iou = []

    def update(self, *inputs):
        y_pred = np.where(inputs[0].asnumpy() > 0.5, 1., 0.).squeeze(1).astype(np.float32)
        y = inputs[1].asnumpy().reshape(-1, self.img_size, self.img_size)
        TP = ((y_pred == 1) & (y == 1)).sum().item()
        FP = ((y_pred == 1) & (y == 0)).sum().item()
        FN = ((y_pred == 0) & (y == 1)).sum().item()
        self.iou.append(TP / (TP + FP + FN))

    def eval(self):
        iou = sum(self.iou) / len(self.iou)
        return iou

class Recall(train.Metric):
    def __init__(self, batch_size=4, num_class=1, img_size=352):
        super(Recall, self).__init__()
        self.batch_size = batch_size
        self.num_class = num_class
        self.img_size = img_size

    def clear(self):
        self.recall = []

    def update(self, *inputs):
        y_pred = np.where(inputs[0].asnumpy() > 0.5, 1., 0.).squeeze(1).astype(np.float32)
        y = inputs[1].asnumpy().reshape(-1, self.img_size, self.img_size)
        TP = ((y_pred == 1) & (y == 1)).sum().item()
        FN = ((y_pred == 0) & (y == 1)).sum().item()
        self.recall.append(TP / (TP + FN))

    def eval(self):
        recall = sum(self.recall) / len(self.recall)
        return recall

class Precision(train.Metric):
    def __init__(self, batch_size=4, num_class=1, img_size=352):
        super(Precision, self).__init__()
        self.batch_size = batch_size
        self.num_class = num_class
        self.img_size = img_size

    def clear(self):
        self.precision = []

    def update(self, *inputs):
        y_pred = np.where(inputs[0].asnumpy() > 0.5, 1., 0.).squeeze(1).astype(np.float32)
        y = inputs[1].asnumpy().reshape(-1, self.img_size, self.img_size)
        TP = ((y_pred == 1) & (y == 1)).sum().item()
        FP = ((y_pred == 1) & (y == 0)).sum().item()
        self.precision.append(TP / (TP + FP))

    def eval(self):
        precision = sum(self.precision) / len(self.precision)
        return precision