import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
from PIL import Image
import sys
from mlp_model import mlp_model

BATCH_SIZE = 1


def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


@flow.global_function("predict")
def eval_job(
        images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
        labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = mlp_model(images, labels, train=False)
    return logits


def main():
    if len(sys.argv) != 2:
        return
    flow.load_variables(flow.checkpoint.get("./mlp_model"))

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32))
    print(logits)

    prediction = np.argmax(logits, 1)
    print("prediction: {}".format(prediction[0]))


if __name__ == '__main__':
    # arr = [[-113.70384, -59.14641, -150.6357, -96.326225, -0.6016015, -101.61007, -144.65833, -0.23799387, -143.88626, -34.610348]]
    main()
