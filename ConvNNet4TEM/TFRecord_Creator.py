from glob import glob
import tensorflow as tf
import argparse
import os
import re

#TFRecord processing
def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image_path, mask_path, image_buffer, mask_buffer, height, width, channels):
    
    feature = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/class/mask':  _bytes_feature(tf.compat.as_bytes(mask_buffer)),
        'image/image_filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(image_path))),
        'image/mask_filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(mask_path))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))
    }
    
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

def _process_image(image_filename, mask_filename):
    # Read the image file.
    image_data = open(image_filename, 'rb').read()    
    mask_data = open(mask_filename, 'rb').read()
    # Decode PNGs
    image = tf.io.decode_png(image_data)
    mask = tf.io.decode_png(mask_data)

    #Check for image and its mask
    assert len(image.shape) == 3
    #Image parameters
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_channels  = image.shape[2]

    #Mask parameters
    mask_height = mask.shape[0]
    mask_width = mask.shape[1]
    mask_channels  = mask.shape[2]

    #Check if image and mask dimensions match
    assert image_height == mask_height
    assert image_width == mask_width

    return image_data, mask_data, image_height, image_width, image_channels, mask_height, mask_width, mask_channels

def natural_sort(sort_list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    sort_list.sort(key=alphanum_key)
    return sort_list

#Create dictionaries matching images with masks
def dict_creator(im_dir):
    images = glob(os.path.join(im_dir, 'images_files', '*.png'))
    masks = glob(os.path.join(im_dir, 'masks_files', '*.png'))
    #Naturally sort images so that they will be in correct order when you will assemble them in prediction code
    images = natural_sort(images)
    masks = natural_sort(masks)
    Image_mask_dict = {}

    #Matching shuffled images with its masks,
    for image, mask in zip(images, masks):
        if os.path.basename(image) == os.path.basename(mask):
            Image_mask_dict[image] = mask
    
    #Dictionary in form of its items so that looping will be easier
    Image_mask_dict = Image_mask_dict.items()
    return Image_mask_dict

def tfrecord_creation(dataset, tfrecord_label, tile_size, outdir):
    #Keeping track of previous filename
    last_value = None
    im_counter = 0
    #Loop over a dataset
    for data in dataset:
        #Get image path
        image_path = data[0]
        #Get mask path
        mask_path = data[1]
        #Basename
        basename = os.path.splitext(os.path.basename(image_path))[0]
        org_file_name = basename.split(':')[0]
        #Output of _process_image
        image_buffer, mask_buffer, image_height, image_width, image_channels, mask_height, mask_width, mask_channels = _process_image(image_path, mask_path)
        
        #Check for correctness of sizes
        if (image_height != tile_size or image_width != tile_size or mask_height != tile_size or mask_width != tile_size):
            continue
        else:
            #Basecase
            if im_counter == 0:
                total_counter = 0
                output_file  = os.path.join(outdir, tfrecord_label, '%d_%s_%s_.tfrecord' % (tile_size, tfrecord_label, org_file_name))
                writer = tf.io.TFRecordWriter(output_file)
            #When filename changes, time to create a new tfrecord
            elif last_value != org_file_name:
                print('%s has %d of %s tiles' % (last_value, im_counter, tfrecord_label))
                writer.close()
                im_counter = 0
                output_file  = os.path.join(outdir, tfrecord_label, '%d_%s_%s_.tfrecord' % (tile_size, tfrecord_label, org_file_name))
                writer = tf.io.TFRecordWriter(output_file)

            #Write to tfrecord your examples 
            example = _convert_to_example(image_path, mask_path, image_buffer, mask_buffer, image_height, image_width, image_channels)
            writer.write(example.SerializeToString())
            im_counter += 1
            total_counter += 1
            #Save last processed filename
            last_value = org_file_name

    print('%s has %d of %s tiles' % (last_value, im_counter, tfrecord_label))
    print('Total amount of %s tiles is: %s' % (tfrecord_label, total_counter))

def main():
    
    #Getting arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, required=True)
    parser.add_argument('--validdir', type=str, required=True)
    parser.add_argument('--testdir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()

    #Casting arguments
    traindir = args.traindir
    validdir = args.validdir
    testdir = args.testdir
    outdir = args.outdir
    tile_size = args.size

    #Check outdir existence
    CHECK_FOLDER = os.path.isdir(outdir)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(outdir)
        print("Created folder: ", outdir)
        for label in ['train', 'valid', 'test']:
            label_outdir = os.path.join(outdir, label)
            os.makedirs(label_outdir)
            print("Created folder: ", label_outdir)

    #Creation of matching dictionaries and TFRecord creator
    try:
        Train_dict = dict_creator(traindir)
        tfrecord_creation(Train_dict, 'train', tile_size, outdir)
    except:
        print("Wasnt able to process VALIDATION dataset")

    try:
        Valid_dict = dict_creator(validdir)
        tfrecord_creation(Valid_dict, 'valid', tile_size, outdir)
    except:
        print("Wasnt able to process VALIDATION dataset")

    try:
        Test_dict = dict_creator(testdir)
        tfrecord_creation(Test_dict, 'test', tile_size, outdir)
    except:
        print("Wasnt able to process TESTING dataset")


if __name__ == "__main__":
    main()
