#
# Copyright (c) 2021, Alexander Zhurkevich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from glob import glob
import tensorflow as tf
import argparse
import os
import csv
import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras

#Model related stuff
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        y_true = y_true < 0.5
        y_pred = y_pred < 0.5
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def get_model(size, kernel_size):
    inputs = keras.Input((size, size, 1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(16, (kernel_size, kernel_size), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [32, 64, 128, 256]:
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

    for filters in [256, 128, 64, 32, 16]:
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
    model = keras.Model(inputs, outputs)
    return model


#Calculate the average
def Average(lst): 
    return sum(lst) / len(lst) 

#TFRecord processing
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
    name = sample['image/image_filename']
    image = tf.cast(image, tf.float32) / 255
    mask = tf.cast(mask, tf.float32) / 255
    return image, mask

def predictor(model, file, testdir, batch_size, steps, csv_name):
    #Create your dataset
    files = tf.data.Dataset.list_files(testdir)
    IoU_per_ckpt = []
    Accuracy_per_ckpt = []
    #Loop over dataset file by file
    for filename in files:
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(read_tfrecord)
        dataset = dataset.batch(batch_size)
        
        #Get the metrics
        metrics = model.evaluate(dataset, steps=steps)
        #print(metrics)
        accuracy = metrics[1]
        IoU = metrics[2]
        
        #Check if it is nan and skip
        if str(IoU) == 'nan':
            break
        else:
            #Append IoU only
            Accuracy_per_ckpt.append(accuracy)
            IoU_per_ckpt.append(IoU)
            
    try:
        #Get the average of IoUs
        average_acc = Average(Accuracy_per_ckpt)
        average_IoU = Average(IoU_per_ckpt)
        
        #Appends averages along with ckpt names to csv
        with open(csv_name, 'a') as csv_file:  
            writer = csv.writer(csv_file)
            writer.writerow([file, average_acc, average_IoU])
            csv_file.close()
    except:
        print("Encountered NaN, skipping checkpoint")

#Responsible for mask merger and overall final processing
def Image_Creator(og_image, true_msk, pred_msk, threshold, saving_path, file_basename, contour_csv_name):
    #Get thresholded true/false masks
    temp_msk = true_msk > 0.5
    temp_pred = pred_msk > threshold

    #Get your background (true) mask
    RGB_msk = np.zeros((temp_msk.shape[0],temp_msk.shape[1],3), dtype=np.float32)

    #Get your predicted mask
    RGB_pred = np.zeros((temp_pred.shape[0],temp_pred.shape[1],3), dtype=np.float32)

    #Blue pixels from mitochondria
    RGB_msk[~temp_msk] = [0.,0.,255.]

    #Red pixels for predicted mitochondria
    RGB_pred[~temp_pred] = [255.,0.,0.]

    #Image conversion
    image = cv2.cvtColor(og_image*255, cv2.COLOR_GRAY2RGB)

    #Get alpha channel from true mask
    alpha_msk = np.zeros((temp_msk.shape[0],temp_msk.shape[1],3), dtype=np.float32)
    alpha_msk[~temp_msk] = [1.,1.,1.]

    #Get alpha channel from prediction mask
    alpha_pred = np.zeros((temp_pred.shape[0],temp_pred.shape[1],3), dtype=np.float32)
    alpha_pred[~temp_pred] = [1.,1.,1.]
            
    #Create forground out of two masks
    foreground_msk = cv2.multiply(alpha_msk, cv2.cvtColor(RGB_msk, cv2.COLOR_BGR2RGB))
    foreground_pred = cv2.multiply(alpha_pred, cv2.cvtColor(RGB_pred, cv2.COLOR_BGR2RGB))
            
    #Merge mask and image
    merged_msk = cv2.add(foreground_msk, image)
    merged_pred = cv2.add(foreground_pred, image)

    #Merge pred mask + true mask + image
    final_merged = cv2.add(foreground_msk, merged_pred)
    #Write the merged image
    cv2.imwrite('%sMERGED_%s.jpeg' % (saving_path, file_basename), final_merged, [cv2.IMWRITE_JPEG_QUALITY, 90])

    #Contours prep
    pred_converted = np.uint8(RGB_pred)
    pred_gray = cv2.cvtColor(pred_converted, cv2.COLOR_BGR2GRAY)
    #Canny edge detection
    pred_edge = cv2.Canny(pred_gray, 100, 150)
    #Extract contours
    pred_contours, pred_hierarchy = cv2.findContours(pred_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Uncomment if you would like to see contours visualization
    #contours_visual = cv2.drawContours(pred_converted, pred_contours, -1, (0,255,0), 10)
    #cv2.imwrite('%sCONTOURS_%s.jpeg' % (saving_path, file_basename), contours_visual, [cv2.IMWRITE_JPEG_QUALITY, 90])

    #Appends contour area, arclen along with image names to contour csv
    with open(contour_csv_name, 'a') as contour_csv_file:  
        writer = csv.writer(contour_csv_file)
        #Write contour information
        for contour in pred_contours:
            writer.writerow([file_basename, cv2.contourArea(contour), cv2.arcLength(contour,True)])
        #Close file
        contour_csv_file.close()

#Main
def main():
    #Getting arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_num', type=str, default='0')
    parser.add_argument('--testdir', type=str)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--naming_pattern', type=str)
    parser.add_argument('--csv_name', type=str)
    parser.add_argument('--contour_csv_name', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    #Casting arguments
    GPU_number = args.GPU_num
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_number
    testdir = args.testdir
    tile_size = args.size
    batch_size = args.batch_size
    kernel_size = args.kernel_size
    threshold = args.threshold
    weights_path = args.weights_path
    naming_pattern = args.naming_pattern
    csv_name = args.csv_name
    contour_csv_name = args.contour_csv_name
    outdir = args.outdir

    #How many image along one cordinate to 
    #sum up together to get to desired resolution
    size_diff = 4096 / tile_size
    #How many you actually need to assemble a full image
    needed_amount = size_diff**2
    #Amount of steps to reach a full dataset coverage due to batching
    steps = needed_amount/batch_size
    
    #Create model
    saved_model = get_model(tile_size, kernel_size)
    #Same metrics as training
    metrics = ["acc", iou]

    #Open csv and write metric names
    with open(outdir+csv_name, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Accuracy", "IoU"])
        csv_file.close()
  
    #Open contour csv and write names, area and arclen 
    with open(outdir+contour_csv_name, 'w') as contour_csv_file:  
        writer = csv.writer(contour_csv_file)
        writer.writerow(["Im_name", "Area", "Arclength"])
        contour_csv_file.close()

    #Loop over checkpoints making predictions
    for file in os.listdir(weights_path):
        #print(file)
        if file.startswith(naming_pattern) and file.endswith('.hdf5'):
            print(file)
            #print(weights_path)
            saved_model.load_weights(weights_path+file)
            saved_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
            #print(saved_model)
            predictor(saved_model, file, testdir, batch_size, steps, outdir+csv_name) 
            
    #Open csv for readfinal_test
    with open(outdir+csv_name, 'r') as csv_file:  
        data = csv.reader(csv_file)
        ResultsDict = {}
        for idx, element in enumerate(data):
            print(element)
            #Populate dictionary with ckpt and IoU 
            if (idx == 0):
                continue
            else:
                ResultsDict[element[0]] = round(float(element[2]), 10)
    
    #Get highest preforming weigths based on IoU
    top_weights = max(ResultsDict, key=ResultsDict.get) 
    print("TOP WEIGHTS: "+top_weights)

    ##Load top weights
    saved_model.load_weights(weights_path+top_weights)

    #Create your dataset
    files = tf.data.Dataset.list_files(testdir)
    #Loop over dataset file by file
    for filename in files:
        #print(filename)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(read_tfrecord)
        dataset = dataset.batch(batch_size)
        #Predictions
        predictions = saved_model.predict(dataset, steps=steps)
        #Getting names of files
        file_path = str(filename.numpy(), encoding='ascii')#[:-4]
        file_basename = os.path.basename(file_path)[:-9]
        print(file_basename)
        
        #Keeping track of amount of tiles
        index_counter = 0
        for element in dataset:
            #Get index
            #Get array of images 
            image_array = element[0]
            #Get array of masks
            mask_array = element[1]
            for idx in range(batch_size):
                #Get image by index
                opened_image = image_array[idx].numpy().reshape(tile_size, tile_size)
                #Get mask by index
                opened_mask = mask_array[idx].numpy().reshape(tile_size, tile_size)
                #Getting prediction by index
                opened_prediction = predictions[index_counter].reshape(tile_size, tile_size) #> threshold
                                
                if index_counter == 0:
                    #Reinitialize all arrays
                    final = np.array([])
                    horizontal_im = np.array([])
                    horizontal_msk = np.array([])
                    horizontal_pred = np.array([])
                    if (tile_size == 4096):
                        horizontal_im = opened_image
                        horizontal_msk = opened_mask
                        horizontal_pred = opened_prediction
                        #Finalize results
                        Image_Creator(horizontal_im, horizontal_msk, horizontal_pred, threshold, outdir, file_basename, outdir+contour_csv_name)
                        break
                    else:
                        vertical_im = np.array([])
                        vertical_msk = np.array([])
                        vertical_pred = np.array([])
                        #Put image/mask/pred into its vertical arrays
                        vertical_im = opened_image
                        vertical_msk = opened_mask
                        vertical_pred = opened_prediction
                elif (index_counter % size_diff) != 0:
                    #Concatenate existing vertical array with a new image/mask/pred
                    vertical_im = np.concatenate((vertical_im, opened_image))
                    vertical_msk = np.concatenate((vertical_msk, opened_mask))
                    vertical_pred = np.concatenate((vertical_pred, opened_prediction))
                elif (index_counter % size_diff) == 0:
                    if horizontal_im.size == 0 and horizontal_msk.size == 0 and horizontal_pred.size == 0:
                        #Add to horizontal, reinitialize your vertical
                        horizontal_im = vertical_im
                        horizontal_msk = vertical_msk
                        horizontal_pred = vertical_pred
                        vertical_im = np.array([])
                        vertical_msk = np.array([])
                        vertical_pred = np.array([])
                        vertical_im = opened_image
                        vertical_msk = opened_mask
                        vertical_pred = opened_prediction
                        #print(horizontal_im.shape)
                    else:
                        #When your vertical array is done concatenate with bigger horizontal one
                        horizontal_im = np.concatenate((horizontal_im, vertical_im), axis=1)
                        horizontal_msk = np.concatenate((horizontal_msk, vertical_msk), axis=1)
                        horizontal_pred = np.concatenate((horizontal_pred, vertical_pred), axis=1)
                        vertical_im = np.array([])
                        vertical_msk = np.array([])
                        vertical_pred = np.array([])
                        vertical_im = opened_image
                        vertical_msk = opened_mask
                        vertical_pred = opened_prediction

                #Increment your total per image tile counter
                index_counter += 1
        
        #Final per image assembly
        if horizontal_im.size != 0 and horizontal_msk.size != 0 and horizontal_pred.size != 0 and tile_size < 4096:
            #Finalize your individual horizontal arrays on image/mask/pred basis
            horizontal_im = np.concatenate((horizontal_im, vertical_im), axis=1)
            horizontal_msk = np.concatenate((horizontal_msk, vertical_msk), axis=1)
            horizontal_pred = np.concatenate((horizontal_pred, vertical_pred), axis=1)

            #Finalize results
            Image_Creator(horizontal_im, horizontal_msk, horizontal_pred, threshold, outdir, file_basename, outdir+contour_csv_name)
       
#Call your main     
main()
