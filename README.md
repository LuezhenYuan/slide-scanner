# slide-scanner
Imaging processing and deconvolution on slide scanner images

## Instlal packages
Install openslide python
sudo apt-get install openslide-tools
conda create --no-default-packages --name slide_py3 python=3.9

source activate slide_py3

conda install -n slide_py3 numpy scikit-image pillow
conda install -c conda-forge -n slide_py3 nd2reader imreg_dft tifffile
pip install openslide-python flask

## run on the slurm cluster
Example code:

srun --cpus-per-task=60 --ntasks=1 --partition=hourly --output=slurm_output-old-2hr-fibronectin.log --error=slurm_output.err python deconvolution_zstack_slide_parallel.py $DATAPATH/old-2hr-fibronectin-actin-aTubulin-HP1a-2_20220503162716264/full-zstack-ch1.tif PSF-BW-emission-DAPI-461nm-325nmpx-2umzstep-20x-NA0o8.tif 60

