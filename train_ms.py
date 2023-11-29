import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from datetime import datetime
import time

import mindspore as ms
import mindspore.nn as nn

from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DUCK_NET_MS import DuckNet
from ModelArchitecture.Matrix import PixelAccuracy, IntersectionOverUnion, Recall, Precision
from ImageLoader.ImageDataset_ms import get_whole_dataset, create_dataset

img_size = 352
num_class = 1
seed_value = 58800
batch_size = 4
folder_path = "/data4/zk/project/DuckNet/kvasir/"  # Add the path to your data directory

dataset_type = 'kvasir' # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
learning_rate = 1e-4
filters = 17 # Number of filters, the paper presents the results with 17 and 34
EPOCHS = 600
min_loss_for_saving = 0.2

ct = datetime.now()

model_type = "DuckNet"
progress_root = "./ProgressFull/"
best_ckpt_dir = "./BestCheckpoint"  # 最佳模型保存路径
if not os.path.exists(best_ckpt_dir): os.mkdir(best_ckpt_dir)
if not os.path.exists(progress_root): os.makedirs(progress_root)
progress_path = progress_root + dataset_type + '_progress_csv_' + model_type + '_filters_' + str(filters) +  '_' + str(ct) + '.csv'
progressfull_path = progress_root + dataset_type + '_progress_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.txt'
plot_path = progress_root + dataset_type + '_progress_plot_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.png'

split_folder_path = get_whole_dataset(folder_path=folder_path, img_size=img_size, seed_value=seed_value)
train_dataset = create_dataset(split_folder_path["train"], batch_size=batch_size, augment=True, shuffle=True, seed_value=seed_value)
step_size_train = train_dataset.get_dataset_size()
val_dataset = create_dataset(split_folder_path["val"], batch_size=batch_size, augment=False, shuffle=False, seed_value=seed_value)
test_dataset = create_dataset(split_folder_path["test"], batch_size=batch_size, augment=False, shuffle=False, seed_value=seed_value)

# Creating the model
network = DuckNet(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=num_class, starting_filters=filters)
optimizer = nn.RMSProp(params=network.trainable_params(), learning_rate=learning_rate)
loss_fn = nn.DiceLoss()
model = ms.train.Model(network, loss_fn, optimizer, metrics={"pixel accuracy": PixelAccuracy(num_class=num_class),
                                                             "IoU": IntersectionOverUnion(num_class=num_class),
                                                             "recall": Recall(num_class=num_class),
                                                             "precision": Precision(num_class=num_class)})

train_acc_tool = PixelAccuracy(num_class=num_class)
train_iou_tool = IntersectionOverUnion(num_class=num_class)
train_recall_tool = Recall(num_class=num_class)
train_precision_tool = Precision(num_class=num_class)

def forward_fn(inputs, targets):
    logits = network(inputs)
    # pred = np.where(logits.asnumpy() > 0.5, 1., 0.)
    pred = ms.numpy.where(logits > 0.5, 1., 0.)
    train_acc_tool.update(pred, targets)
    train_iou_tool.update(pred, targets)
    train_recall_tool.update(pred, targets)
    train_precision_tool.update(pred, targets)

    loss = loss_fn(logits, targets)

    return loss
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
def train_step(inputs, targets):
    loss, grads = grad_fn(inputs, targets)
    optimizer(grads)

    return loss


# 创建迭代器
data_loader_train = train_dataset.create_tuple_iterator(num_epochs=EPOCHS)

# 开始循环训练
print("Start Training Loop ...")

best_iou = 0

for epoch in range(EPOCHS):
    losses = []
    network.set_train()

    epoch_start = time.time()

    train_acc_tool.clear()
    train_iou_tool.clear()
    train_recall_tool.clear()
    train_precision_tool.clear()
    # 为每轮训练读入数据
    for i, (images, labels) in enumerate(data_loader_train):
        images = images.astype(ms.float32)
        labels = labels.astype(ms.int32)
        loss = train_step(images, labels)
        losses.append(loss)

    # 每个epoch结束后，验证准确率
    matrix = model.eval(val_dataset)

    epoch_end = time.time()
    epoch_seconds = (epoch_end - epoch_start)
    step_seconds = epoch_seconds / step_size_train

    print("-" * 20)
    print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f] | "
          "Train pixel Acc: [%5.3f] | "
          "Train IoU: [%5.3f] | "
          "Train R: [%5.3f] | "
          "Train P: [%5.3f] | "
          "Val pixel Acc: [%5.3f] | "
          "Val IoU: [%5.3f] | " 
          "Val R: [%5.3f] | "
          "Val P: [%5.3f] " % (
        epoch + 1, EPOCHS, sum(losses) / len(losses), train_acc_tool.eval(), train_iou_tool.eval(),
        train_recall_tool.eval(), train_precision_tool.eval(), matrix["pixel accuracy"], matrix["IoU"], matrix["recall"], matrix["precision"]
    ))
    print("epoch time: %.3f s, per step time: %.3f s" % (
        epoch_seconds, step_seconds
    ))

    if matrix["IoU"] > best_iou:
        best_iou = matrix["IoU"]
        ms.save_checkpoint(network, os.path.join(best_ckpt_dir, "BestDuckNet.ckpt"))
        print("The best IoU achieved so far is: [%5.3f], the weight has been saved" % (matrix["IoU"]))
        matrix_test = model.eval(test_dataset)
        print("Test Acc: [%5.3f] | Test IoU: [%5.3f] | Test R: [%5.3f] | Test P: [%5.3f]" % (
            matrix_test["pixel accuracy"],
            matrix_test["IoU"],
            matrix_test["recall"],
            matrix_test["precision"]))

print("=" * 80)
print(f"End of validation the best mIoU is: {best_iou: 5.3f}", flush=True)