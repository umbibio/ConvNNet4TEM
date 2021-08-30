# Drug prediction with Electromagnetic Data
Fast and efficient ML semantic segmentation pipeline 

This set of programs and instructions will perform tiling of electromagnetic images, TFRecords creation, semantic segmentation model training, evalations and visualizations. 

# Export [QuPath](https://qupath.github.io/) masks
Start by opening your QuPath project, navigate to Automate -> Show script editor. Copy over contents of QuPath_Export.txt and click Run. This script will create additional folder called export with mitochondria segmentation masks inside. 

# Prepare masks for ML pipeline
Carefully follow Post_Export.txt bash script and edit commands according to where you would like to store images and masks.

# ML Pipeline
## 1. **Installation**:


## 2. **Tiling**:

Tile your images along with corresponding masks.
To run **Tiling.py** in my containder you can: `docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) -w /mnt -v /your_data:/mnt em python Tiling.py --train 80 --valid 10 --test 10 --threads 12 --size 256 --overlap 128 --format png --quality 100 --outdir "/mnt/outdir/" --imdir "/mnt/path_to_images/*.tif" --mskdir "/mnt/path_to_masks/*.png`.

Arguments:
  - `--train` type=int. Percentage of dataset that goes in training (ex. 84).
  - `--valid` type=int. Percentage of dataset that goes in validation (ex. 8).
  - `--test` type=int. Percentage of dataset that goes in testing (ex. 8).
  - `--threads` type=int. How many CPU threads you would like to use (ex. 8).
  - `--overlap` type=int. You can tile your slide with overlap of `N` pixels. **Remember!!!**: the formula for overlap: `tile size + 2 * overlap`, so if you want tiles of size 512x512, you need to pass 256 as `--size` argument and 128 as `--overlap` argument. If you want more info [OpenSlide docu](https://openslide.org/api/python/). Example: 64.  
  - `--size` type=int. Tile size. Example: 512 (512x512).  
  - `--format` type=str. Format (extention) of your tiles (ex png, jpeg). Highly recommend png, otherwise code needs some internal changes.
  - `--quality` type=str. Quality of your tiles (ex 100). Highly recommend 100.
  - `--bounds` type=bool. No need to pass this argument, default is already set.
  - `--outdir` type=str. Output directory where you would like to see you tiles. (ex "/home/username/Documents/tiled_images")
  - `--imdir` type=str. Directory with your images that you would like to tile. (ex "/home/username/Documents/images/*tif")
  - `--mskdir` type=str. Directory with your masks that you would like to tile. (ex "/home/username/Documents/masks/*png")

## 2. **TFRecord_Creator**:

Create TFRecords for faster data throughput when training:
To run **TFRecord_Creator.py** in my containder you can: `docker run --rm --gpus "device=0" -it -u $(id -u ${USER}):$(id -g ${USER}) -w /mnt -v /your_data:/mnt em python TFRecord_Creator.py --size 512 --traindir "/mnt/path_to/train512/" --validdir "/mnt/path_to/valid512/" --test "/mnt/path_to/test512/" --outdir "/mnt/path_to/tfrecords512"`.

Arguments:
  - `--traindir` type=str. Directory with your tiled training images.
  - `--validdir` type=str. Directory with your tiled validation images.
  - `--testdir` type=str. Directory with your tiled test images.
  - `--outdir` type=str. Output directory where you would like to see you TFRecords. (ex "/home/username/Documents/tfrecords").
  - `--size` type=int. Your tile size from previous step (ex. 512), no need to worry about overlap here, just use expected tile size.
 
## 3. **Training**:

Train U-net Neural Network architecture.
To run **Unet_NN.py** in my containder you can: `docker run --rm --gpus "device=0" -it -u $(id -u ${USER}):$(id -g ${USER}) -w /mnt -v /data:/mnt em python Unet_NN.py --batch_size 12 --kernel_size 5 --GPU_num '0' --size 512 --train_num 54072 --valid_num 1280 --epochs 100 --size 512 --ckpt_name "Unet_512" --ckpt_save_freq 10 --train_dir "/mnt/path_to/tfrecords512/512_train*.tfrecord" --valid_dir "/mnt/path_to/tfrecords512/512_valid*.tfrecord" --csv_log_name "/mnt/Logs/Training.log" --tensorboard_logs "/mnt/Logs/TB_logs" --MP "Yes"

Arguments:
  - `--batch_size` type=int. Your typical batch size, scaled linearly with multiple GPU. Example: 64. 
  - `--GPU_num` type=str. Which GPUs your want to use, one digit for one particular GPU, multiple for multiple GPUs (comma separated). Examples: '0' (first availbale GPU) or '0,1' (first two GPUs).
  - `--train_num` type=int. Number of training images, was given at the end of **TFRecord_Creator** execution. Example: 54072.
  - `--valid_num` type=int. Number of validation images, was given at the end of **TFRecord_Creator** execution. Example: 1280.
  - `--epochs` type=int. For how many epochs you would like to run. Example: 100.
  - `--size` type=int. Tile size. Example: 512 (512x512).  
  - `--train_dir` type=str. This argument expects train files' `glob` pattern. Example: '/mnt/YOUR_TFRecords/512_train*.tfrecord'
  - `--valid_dir` type=str. This argument expects validation files' `glob` pattern. Example: '/mnt/YOUR_TFRecords/512_valid*.tfrecord'
  - `--ckpt_name` type=str. This is the name pattern with which your model checkpoints will be saved. Example: '/mnt/YOUR_TRAIN_OUTDIR/512_Unet'
  - `--csv_log_name` type=str. This is a log name that will store your training progress information in a csv file, you can also pass filepath with it. Example: '/mnt/YOUR_TRAIN_OUTDIR/Training.log'
  - `--tensorboard_logs` type=str. This is a folder which will have all information needed for [TensorBoard](https://www.tensorflow.org/tensorboard/get_started). If you are not familiar with this tool, I highly suggest checking it out. 
  - `--MP` this argument is for [Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). If you are not familiar with the concept I highly suggest checking it out, it can speed up your training up to 3.3x, you can also fit 2x batch size. Example: 'Yes' or 'No'.
  
## 4. **Evaluations and visualizations**:


