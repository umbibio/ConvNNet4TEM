# Drug prediction with Electromagnetic Data
Fast and efficient ML semantic segmentation pipeline 

This set of programs and instructions will perform tiling of electromagnetic images, TFRecords creation, semantic segmentation model training, evalations and visualizations. 

# Export [QuPath](https://qupath.github.io/) masks
Start by opening your QuPath project, navigate to Automate -> Show script editor. Copy over contents of QuPath_Export.txt and click Run. This script will create additional folder called export with mitochondria segmentation masks inside. 

# Prepare masks for ML pipeline
Carefully follow Post_Export.txt bash script and edit commands according to where you would like to store images and masks.

# ML Pipeline



