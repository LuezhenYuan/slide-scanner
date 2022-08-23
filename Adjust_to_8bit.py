import sys
import tifffile
import numpy as np
from skimage import filters

slide_name = sys.argv[1] # 'full-zstack-ch1-test-maxzproject.tif'
out_name = slide_name.rsplit('.', 1)[0]

m_slide = tifffile.imread(slide_name)

val = filters.threshold_otsu(m_slide)

m_slide_list = m_slide[m_slide>val]
m_slide_list_mean = np.mean(m_slide_list)
m_slide_list_std = np.std(m_slide_list)

# Convert image to uint8 data type
img_raw_min=m_slide.min()
img_raw_max = m_slide_list_mean + 5*m_slide_list_std
if np.sum(m_slide>img_raw_max)==0:
    img_raw_max = m_slide.max()

m_slide_uint8 = 255*1.0/(img_raw_max-img_raw_min) * (m_slide-img_raw_min)
m_slide_uint8[m_slide_uint8>255]=255
m_slide_uint8 = m_slide_uint8.astype(np.uint8)

tifffile.imwrite(out_name+'-uint8.tif',m_slide_uint8, imagej=True)