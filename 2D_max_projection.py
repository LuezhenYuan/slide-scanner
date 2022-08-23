import sys
import tifffile
from util import slide_tif
import numpy as np

slide_name = sys.argv[1] 
out_name = slide_name.rsplit('.', 1)[0]

m_slide = tifffile.imread(slide_name)

m_slide_proj = slide_tif.z_projection_max(m_slide)

tifffile.imwrite(out_name+'-project.tif',np.round(m_slide_proj).astype('uint16'), imagej=True)
