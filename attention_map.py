import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import scipy
import os


# Get原model的AVG POOLING之前的层和最后层的Weights
def get_ResNet(model):
    # get AMP layer weights
    w1 = model.layers[-2].get_weights()[0]
    w2 = model.layers[-1].get_weights()[0]
    # extract wanted output
    ResNet_model = Model(inputs=model.input,
                         outputs=(model.layers[-4].output,
                                  model.layers[-1].output))
    return ResNet_model, w1, w2


# 计算加权平均
def ResNet_CAM(img_path, model, w1, w2, size=512, channels=2048):
    last_conv_output, pred = model.predict(img_path)
    last_conv_output = np.squeeze(last_conv_output)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1)
    amp_layer_weights = np.dot(w1, w2)[:, 0]
    final_output = -np.dot(mat_for_mult.reshape((size*size, channels)),
                           amp_layer_weights).reshape(size, size)
    return final_output


# 制图
def plot_ResNet_CAM(img_path, model, w1, w2, index):
    im = np.squeeze(img_path)
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.imshow(im)
    plt.title(f'Original Image {index}')
    plt.axis('off')
    ax2 = plt.subplot(1, 2, 2)
    p2 = ax2.imshow(im, alpha=0.4)
    CAM = ResNet_CAM(img_path, model, w1, w2)
    p3 = ax2.imshow(CAM, cmap='jet', alpha=0.6)
    plt.title('Attention Map (Model 1)')
    plt.axis('off')
    colorAx = fig.add_axes([0.95, 0.25, 0.02, 0.5])
    cb = plt.colorbar(p3, cax=colorAx, extend='max')
    return CAM.min(), CAM.max()


if __name__ == '__main__':
    pass
