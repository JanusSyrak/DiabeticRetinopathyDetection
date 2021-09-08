def run_test(model_name, lr, batch_size, threshold):
    import keras
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping
    import os

    from ModelFactory import buildModel, get_input_size
    from MetricsCallback import Histories
    from StaticDataAugmenter import super_directory_size

    # Set up test_values
    model_name = model_name
    lr = lr
    input_size = get_input_size(model_name)
    batch_size = batch_size

    # Initialize data
    trainDirectory = 'RetinaMasks_01_BT/train'
    valDirectory = 'RetinaMasks_01_BT/val'
    testDirectory = 'RetinaMasks_01_BT/test'

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        trainDirectory,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        valDirectory,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        testDirectory,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical')


    # Create model
    model = buildModel(model_name, learning_rate=lr)

    # Initialize metrics
    file_name = model_name + '_' + str(lr) + '_' + "t" + str(threshold)
    # Generalize t127.txt
    results_path = os.path.join('results', file_name + ".txt")
    metrics = Histories(validation_generator, results_path)
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=20,
                       verbose=0,
                       mode='auto',
                       baseline=None,
                       restore_best_weights=False)

    saved_models_path = os.path.join('saved_models', file_name)
    saved_models_path = saved_models_path + "_{epoch:02d}.hdf5"
    mcp = keras.callbacks.ModelCheckpoint(saved_models_path,
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=False,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)


    # Train model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=super_directory_size(trainDirectory) // batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=super_directory_size(valDirectory) // batch_size,
        callbacks=[metrics, es, mcp],
        workers=8,
        shuffle=False,
        verbose=0
    )