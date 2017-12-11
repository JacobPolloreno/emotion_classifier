import click
import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, Flatten
from keras.layers import Input, Reshape, Dense, Dropout
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import get_data, create_bottleneck_feats

_LABELS = ['angry', 'disgust', 'fear', 'happy',
           'sad', 'suprise', 'neutral']
_LOG_DIR = 'log/'


# MODELS

def basic_cnn(lr: float, dropout_rate: float):
    inputs = Input((48, 48))
    reshaped_inputs = Reshape((48, 48, 1))(inputs)
    conv1 = Conv2D(kernel_size=[3, 3], strides=1, filters=32,
                   padding='same', activation='relu')(reshaped_inputs)
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = Conv2D(kernel_size=[3, 3], strides=1, filters=64,
                   padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    conv3 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                   padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=[2, 2], strides=2)(conv3)
    flatten = GlobalAveragePooling2D()(pool3)
    dense = Dense(units=1024)(flatten)
    dropout = Dropout(rate=dropout_rate)(dense)
    logits = Dense(len(_LABELS), activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=logits)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def basic_cnn_with_batch_norm(lr: float):
    inputs = Input((48, 48))
    reshaped_inputs = Reshape((48, 48, 1))(inputs)
    conv1 = Conv2D(kernel_size=[3, 3], strides=1, filters=32,
                   padding='same')(reshaped_inputs)
    bn1 = BatchNormalization()(conv1)
    conv1_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=32,
                     padding='same')(bn1)
    bn1_2 = BatchNormalization()(conv1_2)
    act_1 = Activation('relu')(bn1_2)
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_1)
    dropout_1 = Dropout(rate=0.4)(pool1)

    conv2 = Conv2D(kernel_size=[3, 3], strides=1, filters=64,
                   padding='same')(dropout_1)
    bn2 = BatchNormalization()(conv2)
    conv2_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=64,
                     padding='same')(bn2)
    bn2_2 = BatchNormalization()(conv2_2)
    act_2 = Activation('relu')(bn2_2)
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_2)
    dropout_2 = Dropout(rate=0.4)(pool2)

    conv3 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                   padding='same')(dropout_2)
    bn3 = BatchNormalization()(conv3)
    conv3_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                     padding='same')(bn3)
    bn3_2 = BatchNormalization()(conv3_2)
    act_3 = Activation('relu')(bn3_2)
    pool3 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_3)

    flatten = GlobalAveragePooling2D()(pool3)
    dense = Dense(units=1024)(flatten)
    # dropout = Dropout(rate=0.4)(dense)
    logits = Dense(len(_LABELS), activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=logits)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def deep_cnn(lr: float, dropout_rate: float):
    inputs = Input((48, 48))
    reshaped_inputs = Reshape((48, 48, 1))(inputs)
    conv1 = Conv2D(kernel_size=[5, 5], strides=1, filters=32,
                   padding='same')(reshaped_inputs)
    bn1 = BatchNormalization()(conv1)
    conv1_2 = Conv2D(kernel_size=[5, 5], strides=1, filters=32,
                     padding='same')(bn1)
    bn1_2 = BatchNormalization()(conv1_2)
    act_1 = Activation('relu')(bn1_2)
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_1)
    dropout_1 = Dropout(rate=dropout_rate)(pool1)

    conv2 = Conv2D(kernel_size=[5, 5], strides=1, filters=64,
                   padding='same')(dropout_1)
    bn2 = BatchNormalization()(conv2)
    conv2_2 = Conv2D(kernel_size=[5, 5], strides=1, filters=64,
                     padding='same')(bn2)
    bn2_2 = BatchNormalization()(conv2_2)
    act_2 = Activation('relu')(bn2_2)
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_2)
    dropout_2 = Dropout(rate=dropout_rate)(pool2)

    conv3 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                   padding='same')(dropout_2)
    bn3 = BatchNormalization()(conv3)
    conv3_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                     padding='same')(bn3)
    bn3_2 = BatchNormalization()(conv3_2)
    act_3 = Activation('relu')(bn3_2)
    pool3 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_3)
    dropout_3 = Dropout(rate=dropout_rate)(pool3)

    conv4 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                   padding='same')(dropout_3)
    bn4 = BatchNormalization()(conv4)
    conv4_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=128,
                     padding='same')(bn4)
    bn4_2 = BatchNormalization()(conv4_2)
    act_4 = Activation('relu')(bn4_2)
    pool4 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_4)
    dropout_4 = Dropout(rate=dropout_rate)(pool4)

    flatten = GlobalAveragePooling2D()(pool4)
    dense = Dense(units=1024)(flatten)
    logits = Dense(len(_LABELS), activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=logits)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def fully_conv(lr: float, dropout_rate: float):
    inputs = Input((48, 48))
    reshaped_inputs = Reshape((48, 48, 1))(inputs)
    conv1 = Conv2D(kernel_size=[3, 3], strides=1, filters=56,
                   padding='same')(reshaped_inputs)
    bn1 = BatchNormalization()(conv1)
    conv1_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=56,
                     padding='same')(bn1)
    bn1_2 = BatchNormalization()(conv1_2)
    act_1 = Activation('relu')(bn1_2)
    pool1 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_1)
    dropout_1 = Dropout(rate=dropout_rate)(pool1)

    conv2 = Conv2D(kernel_size=[3, 3], strides=1, filters=28,
                   padding='same')(dropout_1)
    bn2 = BatchNormalization()(conv2)
    conv2_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=28,
                     padding='same')(bn2)
    bn2_2 = BatchNormalization()(conv2_2)
    act_2 = Activation('relu')(bn2_2)
    pool2 = MaxPooling2D(pool_size=[2, 2], strides=2)(act_2)
    dropout_2 = Dropout(rate=dropout_rate)(pool2)

    conv3 = Conv2D(kernel_size=[3, 3], strides=1, filters=14,
                   padding='same')(dropout_2)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=[2, 2], strides=2)(bn3)
    conv3_2 = Conv2D(kernel_size=[3, 3], strides=1, filters=7,
                     padding='same')(pool3)
    dropout_3 = Dropout(rate=dropout_rate)(conv3_2)
    flatten = GlobalAveragePooling2D()(dropout_3)
    logits = Activation('softmax')(flatten)

    model = Model(inputs=inputs, outputs=logits)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def vgg16_transfer(lr: float, dropout_rate: float):

    # Input are precomputed convultions
    inputs = Input((512, ))
    bn1 = BatchNormalization()(inputs)
    dense = Dense(units=400, activation='relu')(bn1)
    dropout1 = Dropout(rate=dropout_rate)(dense)
    bn2 = BatchNormalization()(dropout1)
    dense_2 = Dense(units=200, activation='relu')(bn2)
    dropout2 = Dropout(rate=dropout_rate / 2)(dense_2)
    logits = Dense(len(_LABELS), activation='softmax')(dropout2)

    model = Model(inputs=inputs, outputs=logits)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


_MODELS = {
    'basic': basic_cnn,
    'basic_with_batch_norm': basic_cnn_with_batch_norm,
    'deep_cnn': deep_cnn,
    'fully_conv': fully_conv,
    'vgg': vgg16_transfer,
}


@click.command()
@click.option('--arch', default='basic', type=click.Choice(_MODELS.keys()),
              help="model architecture to train with")
@click.option('--summary', is_flag=True,
              help="PRINT model summary for architecture")
@click.option('--save_dir', default=_LOG_DIR, help="directory to log to")
@click.option('--run_name', default='1', help="name of run for Tensorboard")
@click.option('--lr', default=0.001, show_default=True,
              help="learning rate for training")
@click.option('--dropout_rate', default=0.3, show_default=True,
              help="dropout rate for training")
@click.option('--num_epochs', default=1000, show_default=True,
              help="number of epochs to train for")
@click.option('--batch_size', default=32, show_default=True,
              help="batch size for training")
@click.option('--patience', default=10, show_default=True,
              help="Early stopping training patience")
@click.option('--validation_split', default=0.1, show_default=True)
@click.option('--test_size', default=0.2, show_default=True,
              help="percentage of date to use for testing/validation")
@click.option('--reduce_lr', is_flag=True,
              help="Reduce learning rate when val loss has stopped improving")
def train_and_evaluate(
        arch: str,
        save_dir: str,
        run_name: str,
        lr: float,
        dropout_rate: float,
        num_epochs: int,
        batch_size: int,
        patience: int,
        validation_split: float,
        test_size: float,
        reduce_lr: bool,
        summary: bool):

    # Check for model summary
    if summary:
        print()
        return _MODELS[arch](0.0, 0.0).summary()

    # Print logging directory
    if save_dir[-1] != '/':
        save_dir += '/'
    run_dir = save_dir + run_name + '/'
    print(f"Logging to \'{run_dir}\'\n")

    # Load training and eval data
    if arch == 'vgg':
        try:
            data = np.load('data/bottleneck/bottleneck_features_vgg16.npz')
        except FileNotFoundError:
            print("Generating bottleneck features...\n\n")
            create_bottleneck_feats()
            data = np.load('data/bottleneck/bottleneck_features_vgg16.npz')

        features, labels = data['features'], data['labels']
    else:
        features, labels = get_data()

    # Split data into train, test, and validation splits
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        features, labels, test_size=test_size, random_state=55)

    # One hot encode labels
    train_labels = to_categorical(train_labels)
    eval_labels = to_categorical(eval_labels)

    # Create callbacks
    callbacks_list = [
        ModelCheckpoint(filepath=run_dir + 'model.h5', verbose=1,
                        monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        TensorBoard(log_dir=run_dir, histogram_freq=1,
                    batch_size=batch_size, write_graph=True,
                    write_images=True)
    ]

    if reduce_lr:
        callbacks_list.append(
            ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=patience // 2, verbose=1))

    model = _MODELS[arch](lr, dropout_rate)
    model.fit(
        x=train_data,
        y=train_labels,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=True,
        callbacks=callbacks_list,
        verbose=2)

    eval_results = model.evaluate(
        x=eval_data,
        y=eval_labels,
        verbose=1)
    print()
    print("test loss - {:.4f}\ttest accuracy - {:.2f}".format(*eval_results))


if __name__ == '__main__':
    train_and_evaluate()
