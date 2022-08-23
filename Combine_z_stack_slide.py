import sys
import tifffile
from util import slide_tif
import numpy as np
import pandas as pd

slide_name = sys.argv[1] # '../old-2hr-fibronectin-actin-aTubulin-HP1a-2_20220503162716264/sample.txt'
# It contains three columns: file_name,channel,z_stack
channel = sys.argv[2] # '1'


slide_name_list = pd.read_csv(slide_name)
file_path = slide_name.rsplit('/', 1)[0]+"/" # should include the last '/'
slide_name_list['file_name'] = file_path + slide_name_list['file_name']

m_slide_stack_DAPI = slide_tif.read_full_tif_list_onechannel(slide_name_list,int(channel))

tifffile.imwrite(file_path+'full-zstack-ch'+channel+'.tif',m_slide_stack_DAPI.astype('uint8'), imagej=True)

m_slide_stack_DAPI_proj = slide_tif.z_projection_max(m_slide_stack_DAPI)
tifffile.imwrite(file_path+'zprojectmax-ch'+channel+'.tif',m_slide_stack_DAPI_proj.astype('uint8'), imagej=True)