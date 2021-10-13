# Organelle prediction from Transmission Electron Microscopy images
Fast and efficient ML semantic segmentation pipeline.

![](Result.jpeg)

This is a collection of scripts to perform semantic segmentation on images. The pipeline consists on the following steps:

- Tiling of images
- TFRecords generation
- Model training
- Evaluation and visualization

## Preprocessing: Export [QuPath](https://qupath.github.io/) masks
Start by opening your QuPath project, navigate to Automate -> Show script editor. Copy over contents of QuPath_Export.txt and click Run. This script will create additional folder called export with mitochondria segmentation masks inside. 

## Preprocessing: Prepare masks for ML pipeline
Carefully follow Post_Export.txt bash script and edit commands according to where you would like to store images and masks.

# ML Pipeline
## 1. Installation:
- To do

## 2. Tiling:

Tile images along with corresponding masks.

```
$ python3 Tiling.py --train 80 --valid 10 --test 10 --threads 12 --size 256 --overlap 128 --format png --quality 100 --outdir "/path_to/outdir/" --imdir "/path_to/images/*.tif" --mskdir "/path_to/masks/*.png"
```

Arguments:
  - `--train` type=int. Percentage of dataset that goes in training (ex. 84).
  - `--valid` type=int. Percentage of dataset that goes in validation (ex. 8).
  - `--test` type=int. Percentage of dataset that goes in testing (ex. 8).
  - `--threads` type=int. How many CPU threads you would like to use (ex. 8).
  - `--overlap` type=int. You can tile your slide with overlap of `N` pixels. **Remember!!!**: the formula for overlap: `tile size + 2 * overlap`, so if you want tiles of size 512x512, you need to pass 256 as `--size` argument and 128 as `--overlap` argument. More info at [OpenSlide docs](https://openslide.org/api/python/). Example: 64.  
  - `--size` type=int. Tile size. Example: 512 (512x512).  
  - `--format` type=str. Format of tiles (ex png, jpeg). Recommend png, otherwise code needs some internal changes.
  - `--quality` type=str. Quality of tiles (ex 100). Recommend 100.
  - `--bounds` type=bool. No need to pass this argument, default is already set.
  - `--outdir` type=str. Output directory where you would like to see you tiles. (ex "/home/username/Documents/tiled_images")
  - `--imdir` type=str. Directory with images that you would like to tile. (ex "/home/username/Documents/images/*tif")
  - `--mskdir` type=str. Directory with masks that you would like to tile. (ex "/home/username/Documents/masks/*png")

## 2. TFRecord_Creator:

Create TFRecords for faster data throughput when training:
```
$ python3 TFRecord_Creator.py --size 512 --traindir "/path_to/train512/" --validdir "/path_to/valid512/" --test "/path_to/test512/" --outdir "/path_to/tfrecords512"
```

Arguments:
  - `--traindir` type=str. Directory with tiled training images.
  - `--validdir` type=str. Directory with tiled validation images.
  - `--testdir` type=str. Directory with tiled test images.
  - `--outdir` type=str. Output directory where you would like to see the TFRecords. (ex "/home/username/Documents/tfrecords").
  - `--size` type=int. Tile size from previous step (ex. 512), no need to worry about overlap here, just use expected tile size.
 
## 3. Training:

Train U-net Neural Network architecture.
```
$ python3 Unet_NN.py --batch_size 12 --kernel_size 5 --GPU_num '0' --size 512 --train_num 54072 --valid_num 1280 --epochs 100 --size 512 --ckpt_name "Unet_512" --ckpt_save_freq 10 --train_dir "/path_to/tfrecords512/512_train*.tfrecord" --valid_dir "/path_to/tfrecords512/512_valid*.tfrecord" --csv_log_name "/path_to/Logs/Training.log" --tensorboard_logs "/path_to/Logs/TB_logs" --MP "Yes"
```

Arguments:
  - `--batch_size` type=int. Typical batch size, scaled linearly with multiple GPU. Example: 64.
  - `--kernel_size` type=int. Kernel size of convolutions layers. Example: 5.
  - `--GPU_num` type=str. Which GPUs to use, one digit for one particular GPU, multiple for multiple GPUs (comma separated). Examples: '0' (first availbale GPU) or '0,1' (first two GPUs).
  - `--train_num` type=int. Number of training images, was given at the end of **TFRecord_Creator** execution. Example: 54072.
  - `--valid_num` type=int. Number of validation images, was given at the end of **TFRecord_Creator** execution. Example: 1280.
  - `--epochs` type=int. Number of epochs for training. Example: 100.
  - `--size` type=int. Tile size. Example: 512 (512x512).  
  - `--train_dir` type=str. This argument expects train files' `glob` pattern. Example: '/path_to/tfrecords512/512_train*.tfrecord'
  - `--valid_dir` type=str. This argument expects validation files' `glob` pattern. Example: '/path_to/tfrecords512/512_valid*.tfrecord'
  - `--ckpt_name` type=str. This is the name pattern with which your model checkpoints will be saved. Example: '/path_to/TRAIN_OUTDIR/512_Unet'.
  - `--ckpt_save_freq` type=int. How often a checkpoint of model is saved, measured in epochs. Example: 10.
  - `--csv_log_name` type=str. This is a log name that will store the training progress information in a csv file, you can also pass filepath with it. Example: '/path_to/TRAIN_OUTDIR/Training.log'
  - `--tensorboard_logs` type=str. This is a folder which will have all information needed for [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). 
  - `--MP` type=str. This argument is for [Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). MP can speed up training up to 3.3x, can also fit 2x batch size. Example: 'Yes' or 'No'.
  
## 4. Evaluation and visualization:

Will run evaluations based on given test dataset, after evaluations of every given checkpoint are done, will select best checkpoint based on IoU metric and make visualizations for every testing image. Visualizations consist of overlaying Ground Truth in Blue and Predicted semantic segmentation in Red. Most likely you would want to select best checkpoint based on validation dataset performance at training step and just passing this single checkpoint with testing dataset for visualization part. 
```
$ python3 Image_assembler.py --testdir "/path_to/tfrecords512/512_test*.tfrecord" --size 512 --weights_path "/path_to/dir_with_checkpoints/" --batch_size 16 --kernel_size 5 --naming_pattern "Unet_512" --csv_name "Unet_512_Eval.csv" --contour_csv_name "Contour.csv" --threshold 0.7 --outdir "/path_to/store_outputs/"
```

Arguments:
  - `--testdir` type=str. This argument expects test files' `glob` pattern. Example: '/path_to/TFRecords/512_valid*.tfrecord'.
  - `--GPU_num` type=str. Which GPUs to use, limited to one GPU. Example: '0'.
  - `--size` type=int. Tile size. Example: 512 (512x512).
  - `--weights_path` type=str. Path to directory of checkpoints' weights or path to one particular checkpoint's weights.
  - `--outdir` type=str. Output directory to store eval and visualization results. Example: '/path_to/Visualizations/'
  - `--naming_pattern` type=str. Naming pattern of checkpoints, corresponds with `--ckpt_name` for previous - training step, no need to include full path since if will look for this pattern in `--weights_path` directory. Example: '512_Unet'
  - `--csv_name` type=str. Name of .csv file that will store eval data. Example: 'eval.csv'
  - `--contour_csv_name` type=str. Name of .csv file that will store area and arclen of predictions contour data. Example: 'contour.csv'
  - `--batch_size` type=int. Typical batch size, Suggestion: use number that is divisible by 8 without any remainder. Example: '16'
  - `--kernel_size` type=int. Kernel size of convolutions layers, must be the same value used during training. Example: 5.
  - `--threshold` type=float. IoU threshold for visualizations. Example: 0.5.
  
