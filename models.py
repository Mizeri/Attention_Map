import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def get_model(model_path, model_type=1):
    if model_type == 1:
        base = keras.applications.resnet_v2.ResNet152V2(
                input_shape=(512, 512, 3),
                include_top=False,
                pooling='avg')
    elif model_type == 2:
        base = keras.applications.inception_v3.InceptionV3(
            input_shape=(512, 512, 3),
            include_top=False,
            pooling='avg')
    elif model_type == 3:
        base = keras.applications.inception_resnet_v2.InceptionResNetV2(
            input_shape=(512, 512, 3),
            include_top=False,
            pooling='avg')
    else:
        raise ValueError('model_type should be 1, 2 or 3')
    x = base.output
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    model = Model(inputs=base.input, outputs=x)
    model.load_weights(model_path)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


if __name__ == '__main__':
    a = get_model('/Users/zhaoqian/Downloads/results_refrac_error_pred/results/model1.h5', 1)
    a.summary()
