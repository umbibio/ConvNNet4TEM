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

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import subprocess
from glob import glob
from multiprocessing import Process, JoinableQueue
import time
import os
import sys
from PIL import Image
import argparse
import random
import numpy as np

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, mask, source, tile_size, overlap, bounds, quality, label):
        Process.__init__(self, name='TileWorker')

        self._queue = queue
        self._mask = mask
        self._overlap = overlap
        self._bounds = bounds
        self._quality = quality
        self._tile_size = tile_size
        self._source = source
        self._label = label


    def run(self):
        #Establish generators
        dz_mask = DeepZoomGenerator(self._mask, self._tile_size, self._overlap, limit_bounds=self._bounds)
        dz_source = DeepZoomGenerator(self._source, self._tile_size, self._overlap, limit_bounds=self._bounds)
        
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break

            #Get the data from DeepZoomImageTiler
            if self._label == 'train':
                source_level, mask_level, address, outfile, formatting, outfile_mask, outfile_90, outfile_mask_90, \
                    outfile_180, outfile_mask_180, outfile_270, outfile_mask_270, outfile_mirror, outfile_mask_mirror, \
                    outfile_flip, outfile_mask_flip = data
            elif self._label == 'valid' or self._label == 'test':
                source_level, mask_level, address, outfile, formatting, outfile_mask = data
            
            if True:
                try:
                    #Get tiles
                    mask_tile = dz_mask.get_tile(mask_level, address)
                    source_tile = dz_source.get_tile(source_level, address)
                    
                    if self._label == 'train':
                        #Check if slide tile is completely black
                        source_extrema = source_tile.convert('L').getextrema()
                        #Check if mask tile is completely white
                        mask_extrema = mask_tile.convert('L').getextrema()

                        if source_extrema != (0, 0) and mask_extrema != (255, 255):
                            #Original tiles
                            source_tile.convert('L').save(outfile, quality=self._quality)
                            mask_tile.convert('L').save(outfile_mask, quality=self._quality)

                            #90 Rotation
                            source_tile.rotate(90).convert('L').save(outfile_90, quality=self._quality)
                            mask_tile.rotate(90).convert('L').save(outfile_mask_90, quality=self._quality)

                            #180 Rotation
                            source_tile.rotate(180).convert('L').save(outfile_180, quality=self._quality)
                            mask_tile.rotate(180).convert('L').save(outfile_mask_180, quality=self._quality)

                            #270 Rotation
                            source_tile.rotate(270).convert('L').save(outfile_270, quality=self._quality)
                            mask_tile.rotate(270).convert('L').save(outfile_mask_270, quality=self._quality)

                            #Mirror
                            source_tile.transpose(method=Image.FLIP_LEFT_RIGHT).convert('L').save(outfile_mirror, quality=self._quality)
                            mask_tile.transpose(method=Image.FLIP_LEFT_RIGHT).convert('L').save(outfile_mask_mirror, quality=self._quality)

                            #Flip top to bottom
                            source_tile.transpose(method=Image.FLIP_TOP_BOTTOM).convert('L').save(outfile_flip, quality=self._quality)
                            mask_tile.transpose(method=Image.FLIP_TOP_BOTTOM).convert('L').save(outfile_mask_flip, quality=self._quality)

                        self._queue.task_done()

                    elif self._label == 'valid' or self._label == 'test':
                        #Original tiles
                        source_tile.convert('L').save(outfile, quality=self._quality)
                        mask_tile.convert('L').save(outfile_mask, quality=self._quality)

                        self._queue.task_done()
                    
                except Exception as e:
                    # print(level, address)
                    print("image %s failed at dz.get_tile for level %f" % (self._mask, mask_level))
                    # e = sys.exc_info()[0]
                    print(e)
                    self._queue.task_done()


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz_source, dz_mask, slide_filename, im_output, msk_output, formatting, queue, label):
        self._dz_source = dz_source
        self._dz_mask = dz_mask
        self._im_output = im_output
        self._msk_output = msk_output
        self._formatting = formatting
        self._queue = queue
        self._processed = 0
        self._count = 0
        self._label = label
        self._slide_filename = slide_filename

    def run(self):
        self._write_tiles()
       

    def _write_tiles(self):
        source_level = self._dz_source.level_count-1
        mask_level = self._dz_mask.level_count-1
        self._count = self._count+1
            
        ########################################
        #tiledir = os.path.join("%s_files" % self._basename, str(level))
        im_dir = os.path.join("%s_files" % self._im_output)
        msk_dir = os.path.join("%s_files" % self._msk_output)
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        if not os.path.exists(msk_dir):
            os.makedirs(msk_dir)
        cols, rows = self._dz_source.level_tiles[source_level]
        cols1, rows1 = self._dz_mask.level_tiles[mask_level]
        #Loop over rows and columns
        for row in range(rows):
            for col in range(cols):
                #Getting names of tiles
                tilename = os.path.join(im_dir, '%s:%d:%d.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask = os.path.join(msk_dir, '%s:%d:%d.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_90 = os.path.join(im_dir, '%s:%d:%d:deg90.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask_90 = os.path.join(msk_dir, '%s:%d:%d:deg90.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_180 = os.path.join(im_dir, '%s:%d:%d:deg180.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask_180 = os.path.join(msk_dir, '%s:%d:%d:deg180.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_270 = os.path.join(im_dir, '%s:%d:%d:deg270.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask_270 = os.path.join(msk_dir, '%s:%d:%d:deg270.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mirror = os.path.join(im_dir, '%s:%d:%d:mirror.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask_mirror = os.path.join(msk_dir, '%s:%d:%d:mirror.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_flip = os.path.join(im_dir, '%s:%d:%d:flip.%s' % (
                                self._slide_filename, col, row, self._formatting))
                tilename_mask_flip = os.path.join(msk_dir, '%s:%d:%d:flip.%s' % (
                                self._slide_filename, col, row, self._formatting))

                if not os.path.exists(tilename):
                    #Train has augmentations, hence way more names
                    if self._label == 'train':
                        self._queue.put((source_level, mask_level, (col, row),
                            tilename, self._formatting, tilename_mask, tilename_90, tilename_mask_90, 
                            tilename_180, tilename_mask_180, tilename_270, tilename_mask_270, 
                            tilename_mirror, tilename_mask_mirror, tilename_flip, tilename_mask_flip))
                    #No augmentations
                    elif self._label == 'valid' or self._label == 'test':
                        self._queue.put((source_level, mask_level, (col, row),
                            tilename, self._formatting, tilename_mask))
                #self._tile_done()
        

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz_source.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, mask, slide, slide_filename, im_output, msk_output, formatting, overlap,
                bounds, quality, threads, tile_size, label):
        
        #Crop your mask and slide
        self.greyscale_source = Image.open(slide)
        self.source_crop = self.greyscale_source.crop((0, 0, 4096, 4096))
        self.greyscale_mask = Image.open(mask)
        self.mask_crop = self.greyscale_mask.crop((0, 0, 4096, 4096))
        
        #Open Image through openslide
        self._source = ImageSlide(self.source_crop)
        self._mask = ImageSlide(self.mask_crop)

        #Other arguments
        self._slide_filename = slide_filename
        self._im_output = im_output
        self._msk_output = msk_output
        self._formatting = formatting
        self._overlap = overlap
        self._bounds = bounds
        self._queue = JoinableQueue(2 * threads)
        self._threads = threads
        self._label = label
        self._tile_size = tile_size

        for _i in range(threads):
            TileWorker(self._queue, self._mask, self._source, self._tile_size, overlap,
                bounds, quality, self._label).start()

    def run(self):
        mask = self._mask
        slide = self._source
        im_output = self._im_output
        msk_output = self._msk_output
        dz_mask = DeepZoomGenerator(mask, self._tile_size, self._overlap, limit_bounds=self._bounds)
        dz_source = DeepZoomGenerator(slide, self._tile_size, self._overlap, limit_bounds=self._bounds)
        source_tiler = DeepZoomImageTiler(dz_source, dz_mask, self._slide_filename, im_output, msk_output, self._formatting, self._queue, self._label)
        source_tiler.run()
        self._shutdown()

    def _shutdown(self):
        for _i in range(self._threads):
            self._queue.put(None)
        self._queue.join()

def tiler(dataset, label, outdir, formatting, overlap, bounds, quality, threads, tile_size):
    #One iteration = one image + mask tiled
    for data in dataset:
        image_path = data[0]
        mask_path = data[1]
        slide_filename = os.path.splitext(os.path.basename(image_path))[0]
        print("processing: " + slide_filename)   
        #Get true tile size again
        true_tile_size = tile_size+(2*overlap) 
        #Creation of folders
        im_output = os.path.join(outdir, label+str(true_tile_size), 'images')
        msk_output = os.path.join(outdir, label+str(true_tile_size), 'masks')
        if os.path.exists(im_output) or os.path.exists(msk_output):
            print("Image %s already tiled" % slide_filename)
            continue
        try:
        	DeepZoomStaticTiler(mask_path, image_path, slide_filename, im_output, msk_output, formatting, overlap, bounds, quality, threads, tile_size, label).run()
        except Exception as e:
        	print("Failed to process file %s, error: %s" % (slide_filename, sys.exc_info()[0]))
        	print(e)    
    print(label+" is done")


def main():
    #Getting arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=84)
    parser.add_argument('--valid', type=int, default=8)
    parser.add_argument('--test', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--format', type=str, default='png')
    parser.add_argument('--quality', type=str, default=100)
    parser.add_argument('--bounds', type=bool, default=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--imdir', type=str)
    parser.add_argument('--mskdir', type=str)
    args = parser.parse_args()
    
    #Casting arguments
    train_percentage = args.train
    valid_percentage = args.valid
    test_percentage = args.test
    threads = args.threads 
    tile_size = args.size
    overlap = args.overlap
    formatting = args.format
    quality = args.quality
    bounds = args.bounds
    outdir = args.outdir
    imdir = args.imdir
    mskdir = args.mskdir

    #Remember! Using overlap forces you to be cautious: tile_size + 2 * overlap
    #For example: if you want a tile size of 512, you can do 256 + 2 * 128. 
    #The overlap is defined as "the number of extra pixels to add to each interior edge of a tile"
    true_tile_size = tile_size+(2*overlap)

    #Getting tif images
    tif_images = sorted(glob(imdir))
    #Getting png masks
    masks = sorted(glob(mskdir))
    #Randomize images
    random.shuffle(tif_images)

    Image_mask_dict = {}

    #Matching shuffled images with its masks,
    for image in tif_images:
        image_filename = os.path.basename(image)
        for mask in masks:
            mask_filename = os.path.basename(mask)
            if mask_filename.startswith(os.path.splitext(image_filename)[0]):
                Image_mask_dict[image] = mask
            else:
                continue

    #Get its items to get list operation on a dictionary
    Image_mask_items = Image_mask_dict.items()

    #Splitting data into trains/valid/test
    train_data = list(Image_mask_items)[0:int(len(Image_mask_items)*train_percentage/100)] 
    valid_data = list(Image_mask_items)[int(len(Image_mask_items)*train_percentage/100):int(len(Image_mask_items)*(train_percentage+valid_percentage)/100)] 
    test_data = list(Image_mask_items)[int(len(Image_mask_items)*(train_percentage+valid_percentage)/100):len(Image_mask_items)] 

    #Labels
    train_label = 'train'
    valid_label = 'valid'
    test_label = 'test'
    
    tiler(train_data, train_label, outdir, formatting, overlap, bounds, quality, threads, tile_size)
    #No overlap for valid, test tiles so use true tile size, read more at true_tile_size.
    tiler(valid_data, valid_label, outdir, formatting, 0, bounds, quality, threads, true_tile_size)
    tiler(test_data, test_label, outdir, formatting, 0, bounds, quality, threads, true_tile_size)

    #Printing amount of original images in every dataset
    print()
    print("Number of original images in train: %d" % len(train_data))
    print("Number of original images in valid: %d" % len(valid_data))
    print("Number of original images in test: %d" % len(test_data))
    print()

    #Printing amount of tiles
    train_im_dir = os.path.join(outdir, train_label+str(true_tile_size), 'images_files')
    valid_im_dir = os.path.join(outdir, valid_label+str(true_tile_size), 'images_files')
    test_im_dir = os.path.join(outdir, test_label+str(true_tile_size), 'images_files')
    try:
        print("Amount of train tiles: %d" % len([f for f in os.listdir(train_im_dir)if os.path.isfile(os.path.join(train_im_dir, f))]))
    except:
        print("Wasnt able to count amount of tiles in TRAINING dataset")
    try:
        print("Amount of valid tiles: %d" % len([f for f in os.listdir(valid_im_dir)if os.path.isfile(os.path.join(valid_im_dir, f))]))
    except:
        print("Wasnt able to count amount of tiles in VALIDATION dataset")
    try:
        print("Amount of test tiles: %d" % len([f for f in os.listdir(test_im_dir)if os.path.isfile(os.path.join(test_im_dir, f))]))
    except:
        print("Wasnt able to count amount of tiles in TESTING dataset")
main()