#!/bin/bash

cd src/dataset
rm -rf facades
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
gunzip facades.tar.gz
tar -xf facades.tar
rm facades.tar
cd ../..
