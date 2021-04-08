import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import scipy


# Get weights from layers just before average pooling and final output
def get_avg_weight(model):
    # get AMP layer weights
    w1 = model.layers[-2].get_weights()[0]
    w2 = model.layers[-1].get_weights()[0]
    # extract wanted output
    res_net_model = Model(inputs=model.input,
                          outputs=(model.layers[-4].output,
                                   model.layers[-1].output))
    return res_net_model, w1, w2


# Merge the weights
def attention_map(img_path, model, w1, w2, size=512, channels=2048):
    last_conv_output, prediction = model.predict(img_path)
    last_conv_output = np.squeeze(last_conv_output)
    mat_for_multi = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1)
    amp_layer_weights = np.dot(w1, w2)[:, 0]
    final_output = -np.dot(mat_for_multi.reshape((size*size, channels)),
                           amp_layer_weights).reshape(size, size)
    return final_output, prediction


# Plot the attention map and the original picture
def plot_attention_map(img_path, model, w1, w2, index, label, size=512, channels=2048, i=1):
    im = np.squeeze(img_path)
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    p1 = ax1.imshow(im)
    plt.title(f'Original Image {index}\nSE = {label * 36.5 * -1:.3f}')
    plt.axis('off')
    ax2 = plt.subplot(1, 2, 2)
    p2 = ax2.imshow(im, alpha=0.5)
    cam, prediction = attention_map(img_path, model, w1, w2, size=size, channels=channels)
    v_min = cam.min()
    v_max = cam.max()
    cam = (cam - v_min) / (v_max - v_min) * 2 - 1
    p3 = ax2.imshow(cam, cmap='RdBu_r', alpha=0.5)
    plt.title(f'Attention Map (Model {i})\nPred = {prediction[0][0] * 36.5 * -1:.3f}')
    plt.axis('off')
    color_ax = fig.add_axes([0.95, 0.25, 0.02, 0.5])
    cb = plt.colorbar(p3, cax=color_ax, extend='both')
    return v_min, v_max


if __name__ == '__main__':
    # TODO
    pass
