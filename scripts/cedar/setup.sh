#!/bin/bash

# Make the script fail whenever something bugs out or an unset variable is used or when a piped command has an error
#set -euo pipefail
#IFS=$'\n\t'

export SCRATCH=${SCRATCH:=$HOME}

function create_load_environment(){    
    b=`pwd` # save the current directory.
    cd $SCRATCH/repos/SSCL

    echo "Creating the environment locally on the compute node."
    module load python/3.7
    virtualenv --no-download $SLURM_TMPDIR/env
    source $SLURM_TMPDIR/env/bin/activate
    # Install the packages that *do* need an internet connection.
    pip install -r scripts/cedar/requirements.txt

    cd $b  # go back to the original directory.
}

function download_required_stuff(){
    b=`pwd` # save the current directory.

    cd $SCRATCH/repos/SSCL

    # Download the datasets to the $SCRATCH/data directory (if not already downloaded).
    python -m scripts.download_datasets --data_dir $SCRATCH/data
    
    # Download the pretrained model weights to the ~/.cache/(...) directory
    # (accessible from the compute node)
    export TORCH_HOME="$SCRATCH/.torch"
    mkdir -p $TORCH_HOME
    python -m scripts.download_pretrained_models --save_dir $TORCH_HOME

    cd $SCRATCH
    # Zip up the data folder (if it isn't already there)
    zip -u -r -v data.zip data

    # 2. Copy your dataset on the compute node
    # IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
    cp --update --verbose $SCRATCH/data.zip -d $SLURM_TMPDIR
    
    cd $SLURM_TMPDIR
    # 3. Eventually unzip your dataset
    unzip -o -u $SLURM_TMPDIR/data.zip -d $SLURM_TMPDIR

    # go back to the original directory.
    cd $b
}

function copy_code(){
   # TODO: copy the code over to slurm_tmpdir, so that we can edit stuff in $SCRATCH
   # OR: checkout a given branch of the repo, something like that.
   git submodule init
   git submodule update
}


# Make the output directory for the slurm files, if not already present
mkdir -p $SCRATCH/slurm_out

HOSTNAME=`hostname`
echo "HOSTNAME is $HOSTNAME"

create_load_environment
download_required_stuff
copy_code
