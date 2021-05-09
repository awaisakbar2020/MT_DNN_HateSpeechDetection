#!/bin/bash
#OAR -q production
#OAR -p cluster='graffiti'
##OAR -p cluster='grvingt' this is a commented line because of double-#
#OAR -l core=2,walltime=24:00:00
#OAR -O oar_job.%jobid%.output
#OAR -E oar_job.%jobid%.error


source ~/.bashrc

set -xv
if [ -f "/home/aakbar/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/home/aakbar/anaconda3/etc/profile.d/conda.sh"
                        CONDA_CHANGEPS1=false conda activate hatespeech
fi


#python train.py
python prepro_std.py
