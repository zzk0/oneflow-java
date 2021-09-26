from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
import numpy as np
import os

BATCH_SIZE = 100
flow.config.enable_legacy_model_io(True)
flow.config.enable_model_io_v2(True)
flow.enable_eager_execution(False)
print(os.getpid())


def mlp_model(images, labels, train=True):
    # [batch_size, image_sizes] -> [batch_size, pixels]
    # reshape = flow.reshape(images, [images.shape[0], -1])
    reshape = flow.flatten(images, start_dim=1)

    # dense, [batch_size, pixels] -> [batch_size, 500]
    initializer1 = flow.random_uniform_initializer(-1 / 28.0, 1 / 28.0)
    hidden = flow.layers.dense(
        reshape,
        500,
        activation=flow.nn.relu,
        kernel_initializer=initializer1,
        bias_initializer=initializer1,
        name="dense1"
    )

    # dense, [batch_size, 500] -> [batch_size, logits]
    initializer2 = flow.random_uniform_initializer(
        -np.sqrt(1 / 500.0), np.sqrt(1 / 500.0)
    )
    logits = flow.layers.dense(
        hidden,
        10,
        kernel_initializer=initializer2,
        bias_initializer=initializer2,
        name="dense2"
    )

    if train:
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return loss
    else:
        return logits


@flow.global_function(type="train")
def train_job(
        images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
        labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        loss = mlp_model(images, labels)

    flow.optimizer.Adam(
        flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    ).minimize(loss)
    return loss


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )

    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, 20, loss.mean()))
    flow.checkpoint.save("./mlp_model")
