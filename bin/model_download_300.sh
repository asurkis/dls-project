#!/bin/bash

cd src/saved
gdown 'https://drive.google.com/uc?id=1mHM9lMp_yUMi1M7DePZZOln_HbCJN7z9'
rm -f facades_generator_300.pth facades_discriminator_300.pth
7z e facades_300.7z
rm facades_300.7z
cd ../..
