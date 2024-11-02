# !/bin/bash

# Install dependencies
brew install libomp open-mpi gcc direnv ossp-uuid cmake

# Export the compiler override for macos
echo export CXX=g++-14 > .envrc
direnv allow .