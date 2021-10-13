# all .pyx files in a folder
import setuptools
from glob import glob


with open("README.md", "r") as fh:
    long_description = fh.read()


import re
VERSIONFILE="ConvNNet4TEM/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setuptools.setup(
    name = 'ConvNNet4TEM',
    version=verstr,
    author="umbibio",
    author_email="noreply@umb.edu",
    description='Semantic segmentation for TEM images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/umbibio/ConvNNet4TEM',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points = {
        'console_scripts': [
            'cnn4tem-tiling=ConvNNet4TEM.Tiling:main',
            'cnn4tem-tfrecord-creator=ConvNNet4TEM.TFRecord_Creator:main',
            'cnn4tem-unet-nn=ConvNNet4TEM.Unet_NN:main',
            'cnn4tem-image-assembler=ConvNNet4TEM.Image_assembler:main']
    },
    install_requires=['opencv-python', 'openslide-python'],
    zip_safe=False
)
