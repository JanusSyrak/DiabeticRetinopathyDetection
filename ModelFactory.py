from keras import applications
from keras import layers
from keras_applications.resnet import ResNet50

input_sizes = {
    "xception":          (299, 299, 3),
    "vgg16"   :          (224, 224, 3),
    "vgg19"   :          (224, 224, 3),
    "resnet"  :          (224, 224, 3),
    "resnetv2":          (224, 224, 3),
    "resnext" :          (224, 224, 3),
    "inceptionv3":       (299, 299, 3),
    "inceptionresnetv2": (299, 299, 3),
    "mobilenet":         (224, 224, 3),
    "mobilenetv2":       (224, 224, 3),
    "densenet":          (224, 224, 3),
    "nasnetlarge":       (331, 331, 3),
    "natnetmobile":      (224, 224, 3),
    }


def buildModel(model_name, learning_rate, tuning='top_only'):
    if model_name == 'xception':
        return buildXception(tuning, learning_rate)
    if model_name == 'vgg16':
        return buildVGG16(tuning, learning_rate)
    if model_name == 'vgg19':
        return buildVGG19(tuning, learning_rate)
    if model_name == 'resnet':
        return buildResNet(tuning, learning_rate)
    if model_name == 'resnetv2':
        return buildresnet50v2(tuning, learning_rate)
    if model_name == 'resnext':
        return buildresnext50(tuning, learning_rate)
    if model_name == 'inceptionv3':
        return buildInceptionV3(tuning, learning_rate)
    if model_name == 'inceptionresnetv2':
        return buildInceptionResNetV2(tuning, learning_rate)
    if model_name == 'mobilenet':
        return buildMobileNet(tuning, learning_rate)
    if model_name == 'mobilenetv2':
        return buildMobileNetV2(tuning, learning_rate)
    if model_name == 'densenet':
        return buildDenseNet(tuning, learning_rate)
    if model_name == 'nasnet':
        return buildNASNet(tuning, learning_rate)
    if model_name == 'alexnet':
        return buildAlexNet(tuning, learning_rate)
    if model_name == 'lenet':
        return buildLeNet(tuning, learning_rate)



def buildXception(tuning, learning_rate):
    model = applications.xception.Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_sizes["xception"],
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildVGG16(tuning, learning_rate):
    model = applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_sizes["vgg16"],
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildVGG19(tuning, learning_rate):
    model = applications.vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_sizes["vgg19"],
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


# Currently only supports ResNet50
def buildResNet(tuning, learning_rate):
    model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_sizes["resnet"],
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnet50(tuning, learning_rate):
    model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnet101(tuning, learning_rate):
    model = applications.resnet.ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnet152(tuning, learning_rate):
    model = applications.resnet.ResNet152(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnet50v2(tuning, learning_rate):
    model = applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)

def buildresnet101v2(tuning, learning_rate):
    model = applications.resnet_v2.ResNet101V2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnet152v2(tuning, learning_rate):
    model = applications.resnet_v2.ResNet152V2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnext50(tuning, learning_rate):
    model = applications.resnext.ResNeXt50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildresnext101(tuning, learning_rate):
    model = applications.resnext.ResNeXt101(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=2)

    return buildTop(model, learning_rate, tuning=tuning)


def buildInceptionV3(tuning, learning_rate):
    model = applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_sizes["inceptionv3"],
        pooling=None,
        classes=5)

    return buildTop(model, learning_rate, tuning=tuning)


def buildInceptionResNetV2(tuning, learning_rate):
    model = applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=5)

    return buildTop(model, learning_rate, tuning=tuning)


def buildMobileNet(tuning, learning_rate):
    model = applications.mobilenet.MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=1e-3,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=5)

    return buildTop(model, learning_rate, tuning=tuning)


def buildMobileNetV2(tuning, learning_rate):
    pass

# Currently only supports DenseNet121
def buildDenseNet(tuning, learning_rate):
    model = applications.densenet.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=5)

    return buildTop(model, learning_rate, tuning=tuning)


# Currently only supports NASNetLarge
def buildNASNet(tuning, learning_rate):
    model = applications.nasnet.NASNetLarge(
        input_shape=None,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=5)

    return buildTop(model, learning_rate, tuning=tuning)


def buildAlexNet(tuning, learning_rate):
    pass


def buildLeNet(tuning, learning_rate):
    pass

# This function needs to take the output of the previous model as output
def buildTop(input_model, learning_rate ,tuning='top_only'):
    from keras.models import Model
    from keras.layers import Dense, Flatten
    from keras.optimizers import adam
    base_model = input_model

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if tuning == 'fine':
        pass
    elif tuning == 'top_only':
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer=adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_input_size(model_name):
    return (input_sizes[model_name][0], input_sizes[model_name][1])