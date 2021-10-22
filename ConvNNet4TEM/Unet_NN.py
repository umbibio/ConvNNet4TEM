import os
import tensorflow as tf
import numpy as np
import itertools
from datetime import datetime
import sys
from PIL import Image
import argparse
from tensorflow.keras import layers


def read_tfrecord(record):
    keys_to_features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/class/mask': tf.io.FixedLenFeature([], tf.string),
        'image/image_filename': tf.io.FixedLenFeature([], tf.string),
        'image/mask_filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string)
    }
    sample =  tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_png(sample['image/encoded'], channels=1)
    mask = tf.io.decode_png(sample['image/class/mask'], channels=1)
    #Normalize
    image = tf.cast(image, tf.float32) / 255
    mask = tf.cast(mask, tf.float32) / 255
    return image, mask


#Get training dataset
def get_batched_train_dataset(BATCH_SIZE, directory_path):
    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except AttributeError:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    pattern=os.path.join(directory_path, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


#Get validation dataset
def get_batched_valid_dataset(BATCH_SIZE, directory_path):
    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except AttributeError:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    pattern=os.path.join(directory_path, "*.tfrecord")
    files = tf.data.Dataset.list_files(pattern)
    dataset = tf.data.TFRecordDataset(filenames=files)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


def my_iou(y_true, y_pred):
    def f(y_true, y_pred):
        #print(y_pred)
        y_true = y_true < 0.5
        y_pred = y_pred < 0.5
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def get_model(size, kernel_size, model_depth):

    filters_list = [2**i for i in range(4, 4 + model_depth)]
    inputs = tf.keras.Input((size, size, 1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(filters_list[0], (kernel_size, kernel_size), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in filters_list[1:]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, (kernel_size, kernel_size), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, (kernel_size, kernel_size), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D((kernel_size, kernel_size), strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in reversed(filters_list):
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, (kernel_size, kernel_size), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, (kernel_size, kernel_size), padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    x = layers.Conv2D(1, 3, padding="same")(x)
    outputs = layers.Activation("sigmoid", dtype='float32', name='predictions')(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--GPU_num', type=str, default='0')
    parser.add_argument('--train_num', type=int, required=True)
    parser.add_argument('--valid_num', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--model_depth', type=int, default=4)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--valid_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, required=True)
    parser.add_argument('--ckpt_save_freq', type=int, default=10)
    parser.add_argument('--csv_log_name', type=str, required=True)
    parser.add_argument('--tensorboard_logs', type=str, required=True)
    parser.add_argument('--MP', type=str, default="Yes")
    parser.add_argument('--summary_only', action="store_true")
    args = parser.parse_args()

    #Which GPU to use
    GPU_number = args.GPU_num
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_number
    #Size of your batch
    batch_size = args.batch_size
    #Kernel size
    kernel_size = args.kernel_size
    #Number of images in train dataset
    train_num = args.train_num
    #Image size
    size = args.size
    #Model depth, number of filters in model
    model_depth = args.model_depth
    #Number of images in valid dataset
    valid_num = args.valid_num
    #Epochs
    EPOCHS = args.epochs
    #Training directory along with glob file pattern
    train_dir = args.train_dir
    #Valid directory along with glob file pattern
    valid_dir = args.valid_dir
    #Chekpoint names with _epoch#.hdf5 in the end
    ckpt_name = args.ckpt_name
    #Checkpoint saving frequency 
    ckpt_save_freq = args.ckpt_save_freq
    #CSV log filename
    csv_log_name = args.csv_log_name
    #Tensorboard logs location
    tensorboard_logs = args.tensorboard_logs
    #Mixed precision 
    MP = args.MP

    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_number
    #Multi-GPU
    strategy = tf.distribute.MirroredStrategy()
    GPUnum = strategy.num_replicas_in_sync
    print ('Number of devices: {}'.format(GPUnum))
    batch_size = batch_size*GPUnum

    if MP == "Yes":
        #Mixed precision
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print('Compute dtype: %s' % tf.keras.mixed_precision.global_policy())
        except AttributeError:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)
            print('Compute dtype: %s' % tf.keras.mixed_precision.experimental.global_policy())

    #Getting steps
    TRAINING_STEPS_PER_EPOCH = train_num//batch_size
    VALIDATION_STEPS_PER_EPOCH = valid_num//batch_size

    with strategy.scope():
        metrics = ["acc", my_iou]
        model = get_model(size, kernel_size, model_depth)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)

        my_callbacks = [
            tf.keras.callbacks.CSVLogger(csv_log_name, separator=',', append=False),
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_graph=True, update_freq=500, profile_batch=0),
            tf.keras.callbacks.ModelCheckpoint('%s_{epoch:08d}.hdf5' % ckpt_name, save_best_only=False, save_weights_only=False, save_freq=TRAINING_STEPS_PER_EPOCH*ckpt_save_freq)]

    model.summary()
    if args.summary_only:
        print("Requested summary alone. Exiting now...\n")
        tf.keras.backend.clear_session()

        return

    def get_training_dataset():
        return get_batched_train_dataset(batch_size, train_dir)

    def get_validation_dataset():
        return get_batched_valid_dataset(batch_size, valid_dir)

    model.fit(get_training_dataset(),
              validation_data=get_validation_dataset(),
              epochs=EPOCHS,
              steps_per_epoch=TRAINING_STEPS_PER_EPOCH, 
              validation_steps=VALIDATION_STEPS_PER_EPOCH, callbacks=my_callbacks)


if __name__ == "__main__":
    main()
