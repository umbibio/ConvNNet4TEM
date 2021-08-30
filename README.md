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
  - `--overlap` type=int. You can tile your slide with overlap of `N` pixels. **Remember!!!**: the formula for overlap: `tile size + 2 * overlap`, so if you want tiles of size 256x256, you need to pass 128 as `--size` argument and 64 as `--overlap` argument. If you want more info [OpenSlide docu](https://openslide.org/api/python/). Example: 64.  
  - `--format` type=str. Format (extention) of your tiles (ex png, jpeg). Highly recommend png, otherwise code needs some internal changes.
  - `--quality` type=str. Quality of your tiles (ex 100). Highly recommend 100.
  - `--bounds` type=bool. No need to pass this argument, default is already set.
  - `--outdir` type=str. Output directory where you would like to see you tiles. (ex "/home/username/Documents/tiled_images")
  - `--imdir` type=str. Directory with your images that you would like to tile. (ex "/home/username/Documents/images/*tif")
  - `--mskdir` type=str. Directory with your masks that you would like to tile. (ex "/home/username/Documents/masks/*png")



