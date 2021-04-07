import os
import matplotlib.pyplot as plt
from preprocessing import data_to_generator
from attention_map import get_ResNet, plot_ResNet_CAM
from models import get_model
import time

# 为了避免报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    img_path = '/Users/zhaoqian/Downloads/results_refrac_error_pred/results/data'
    csv_path = '/Users/zhaoqian/Downloads/results_refrac_error_pred/results/se.csv'
    model_loc = '/Users/zhaoqian/Downloads/results_refrac_error_pred/results'
    test_size = 2
    test_generator, df_test = data_to_generator(img_path, csv_path, test_size, 2021)

    for i in range(1, 4):
        model_path = f'{model_loc}/model{i}.h5'
        print(model_path)
        model = get_model(model_path, i)
        print(model.summary())
        ResNet_model, w1, w2 = get_ResNet(model)
        k = 0
        for j in test_generator:
            try:
                n = df_test.loc[k, 'file_name'][65:69]
                start = time.time()
                if i == 1:
                    a, b = plot_ResNet_CAM(j[0], ResNet_model, w1, w2, n, 512, 2048, i)
                elif i == 2:
                    a, b = plot_ResNet_CAM(j[0], ResNet_model, w1, w2, n, 448, 2048, i)
                elif i == 3:
                    a, b = plot_ResNet_CAM(j[0], ResNet_model, w1, w2, n, 448, 1536, i)
                end = time.time()
                plt.savefig(f'{n}_model{i}.png', bbox_inches='tight')
                print(f'[+]{n}_model{i}.png')
                print(f'vmin of {n}_model{i} is {a}, vmax is {b}')
                print(f'time consumed: {end - start}')
                print('=' * 98)
                plt.close('all')
                k += 1
            except KeyError:
                break
