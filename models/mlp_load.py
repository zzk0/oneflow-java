import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
from PIL import Image
import os
import time

print(os.getpid())


def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


if __name__ == '__main__':
    # sess = flow.serving.InferenceSession()
    # sess.load_saved_model(saved_model_dir="./models", model_version=1)
    # sess.launch()
    # images = load_image("./7.png")
    # tags = np.zeros((1,)).astype(np.int32)
    # logits = sess.run("mlp_inference", Input_14=images, Input_15=tags)
    # print(logits)
    # prediction = np.argmax(logits[0], 1)
    # print("prediction: {}".format(prediction[0]))
    # sess.close()

    # image_files = os.listdir('test_set')
    # print(image_files)

    # sess = flow.serving.InferenceSession()
    # sess.load_saved_model(saved_model_dir="./models", model_version=1)
    # sess.launch()
    # images = np.array([load_image('test_set/' + image_files[i]) for i in range(10)]).reshape(10, 1, 28, 28)
    # tags = np.zeros((10,)).astype(np.int32)
    # logits = sess.run("mlp_inference", image=images)
    # print(logits)
    # prediction = np.argmax(logits[0], 1)
    # print("prediction: {}".format(prediction[0]))
    # sess.close()


    # sess = flow.serving.InferenceSession()
    # sess.load_saved_model(saved_model_dir="./models", model_version=1)
    # sess.launch()
    # image = load_image("./7.png")
    # tag = np.zeros((1,)).astype(np.int32)

    # forwardTimes = 10000
    # cur_time = time.time()
    # for i in range(forwardTimes):
    #     logits = sess.run("mlp_inference", Input_14=image, Input_15=tag)
    # print('It takes {} s to forward {} times'.format(time.time() - cur_time, forwardTimes))
    # print(logits)
    # prediction = np.argmax(logits[0], 1)
    # print("prediction: {}".format(prediction[0]))
    # sess.close()

    sess = flow.serving.InferenceSession()
    sess.load_saved_model(saved_model_dir="./lenet_models", model_version=1)
    sess.launch()
    images = load_image("./7.png")
    tags = np.zeros((1,)).astype(np.int32)
    cur_time = time.time()
    for i in range(100):
        logits = sess.run("mlp_inference", image=images)
    print(time.time() - cur_time)
    print(logits)
    prediction = np.argmax(logits[0], 1)
    print("prediction: {}".format(prediction[0]))
    sess.close()
