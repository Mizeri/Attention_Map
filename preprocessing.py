import glob
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_to_generator(img_path, csv_path, test_size=197, seed=2021):
    # 新建列表存储文件名
    train_files = []
    for img_name in glob.iglob(f'{img_path}/*.tif'):
        train_files.append(img_name)
    # OS和osse表
    df = pd.read_csv(csv_path)
    # 新建一列文件名
    df['file_name'] = 'nan'
    for i in train_files:
        df.loc[df['OS'] == int(i[65:69]), 'file_name'] = i
    # 新建一列label
    df.loc[:, 'label'] = df.loc[:, 'osse'] * (-1)
    # 没找到图的不要了，看看超高度的有多少
    existed = len(df[df['file_name'] != 'nan'])
    total = len(df)
    print('=' * 70)
    print(f'{existed / total * 100:.2f}%的受试者找到了对应的图片({existed} / {total})')
    df = df[df['file_name'] != 'nan']
    positive = len(df[df['osse'] < -10]) / len(df) * 100
    n = len(df)
    print(f'{positive:.2f}%的受试者为超高度近视（>10D), ', len(df[df['osse'] < -10]), f'/ {n}')
    print('=' * 70)
    df.loc[:, 'file_name'] = df['file_name'].astype(str)
    df.loc[:, 'label'] = df['label'].astype(float)
    print(train_files)
    print('=' * 70)
    # 前197位为测试集
    df_test = df.loc[:test_size, :]
    print(df_test)
    print('=' * 70)
    # 图片流，重缩放以及亮度限制
    tester = ImageDataGenerator(
        rescale=1. / 255,
        brightness_range=(0.8, 1.2),
    )

    test_generator = tester.flow_from_dataframe(
        df_test,
        x_col='file_name',
        y_col='label',
        has_ext=True,
        target_size=(512, 512),
        color_mode='rgb',
        class_mode='raw',
        batch_size=1, sort=False,
        shuffle=False, seed=seed
    )
    return test_generator, df_test


if __name__ == '__main__':
    # TODO
    pass
