#! /bin/bash
#!/bin/bash -ex

root=$SCRATCH
downloads=$root/downloads
dest=$root/dest

rm -r -f $downloads && mkdir -p $downloads
rm -f -r $dest && mkdir -p $dest

# Download newer version of CMAKE
# (the one supported by the cluster is ancient!)
wget https://github.com/Kitware/CMake/releases/download/v3.31.0-rc3/cmake-3.31.0-rc3-linux-x86_64.tar.gz -O $downloads/cmake-bin.tar.gz
mkdir -p $downloads/cmake-bin
tar -C $dest -xzf $downloads/cmake-bin.tar.gz
mv $dest/cmake-3.31.0-rc3-linux-x86_64/* $dest
rm -d $dest/cmake-3.31.0-rc3-linux-x86_64

# Purge the pip cache (otherwise updates will fail)
pip cache purge
