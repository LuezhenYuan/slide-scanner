import sys
import tifffile
from util import slide_tif
import numpy as np

slide_name = sys.argv[1] # '../old-2hr-fibronectin-actin-aTubulin-HP1a-2_20220503162716264/full-zstack-ch1.tif'
out_name = slide_name.rsplit('.', 1)[0]

m_slide = tifffile.imread(slide_name)

m_slide_proj = slide_tif.z_projection_max(m_slide)
tifffile.imwrite(out_name+'zprojectmax'+'.tif',m_slide_proj.astype('uint8'), imagej=True)