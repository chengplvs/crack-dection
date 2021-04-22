from models.build import create_modelA, create_modelB, create_modelC
from tensorflow.keras.applications import ResNet152


def create_resnet152(optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy']):
    # 建立卷積基底
    base_model = ResNet152(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))
    model = create_modelA(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model


def create_resnet152B(optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy']):
    # 建立卷積基底
    base_model = ResNet152(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))

    model = create_modelB(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model


def create_resnet152C(optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy']):
    base_model = ResNet152(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))

    # 解凍有節點層
    unfreeze = ['conv5_block3_1_conv', 'conv5_block3_1_bn', 'conv5_block3_2_conv',
                'conv5_block3_2_bn', 'conv5_block3_3_conv', 'conv5_block3_3_bn']

    for layer in base_model.layers:
      if layer.name in unfreeze:
        layer.trainable = True  # 解凍
      else:
        layer.trainable = False  # 其他凍結權重

    #for layer in base_model.layers[-2:]:
        #layer.trainable = True

    model = create_modelC(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model
