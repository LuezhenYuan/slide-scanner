import sys
import tifffile
from util import deconv
import numpy as np
import multiprocessing as mp
from util import deconv
from joblib import Parallel, delayed
from tqdm import tqdm

slide_name = sys.argv[1] # '../old-2hr-fibronectin-actin-aTubulin-HP1a-2_20220503162716264/full-zstack-ch1.tif'
psf_file = sys.argv[2] #'PSF-BW-emission-DAPI-461nm-325nmpx-2umzstep-20x-NA0o8.tif'
#num_div = int(sys.argv[3]) # 3
num_process = int(sys.argv[3]) # 2

if num_process>mp.cpu_count():
    num_process=mp.cpu_count()

out_name = slide_name.rsplit('.', 1)[0]

psf = tifffile.imread(psf_file)
psf = psf*1.0/np.sum(psf) # added
psf = psf.astype('float16')

m_slide = tifffile.imread(slide_name)

## allocate all subimages
imge_nd = m_slide.astype('float16')
#n_div=num_div
n_overlap=50 # number of pixel

(n_z,n_height,n_width) = imge_nd.shape
"""
height_list = np.zeros((n_div,2),'int')
width_list = np.zeros((n_div,2),'int')
for i in range(n_div):
    height_list[i,:]=[int(n_height/n_div*i-n_overlap),int(n_height/n_div*(i+1)+n_overlap)]
    width_list[i,:]=[int(n_width/n_div*i-n_overlap),int(n_width/n_div*(i+1)+n_overlap)]
"""
n_subimg_size=n_overlap*10
if n_subimg_size>n_height or n_subimg_size>n_width:
    n_subimg_size = np.minimum(n_height,n_width)

height_list = np.zeros((round(n_height/n_subimg_size+0.5),2),'int')
width_list = np.zeros((round(n_width/n_subimg_size+0.5),2),'int')
for i in range(height_list.shape[0]):
    height_list[i,:]=[int(n_subimg_size*i-n_overlap),int(n_subimg_size*(i+1)+n_overlap)]

for j in range(width_list.shape[0]):
    width_list[j,:]=[int(n_subimg_size*j-n_overlap),int(n_subimg_size*(j+1)+n_overlap)]

height_list[height_list<0]=0
width_list[width_list<0]=0
height_list[height_list>n_height-1]=n_height-1
width_list[width_list>n_width-1]=n_width-1

# start parallel
"""
pool = mp.Pool(num_process)
results_subimg = [pool.apply(deconv.deconvolution_DAPI, args=(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf)).copy() for i in range(height_list.shape[0]) for j in range(width_list.shape[0])]

pool.close()
"""
results_subimg = Parallel(n_jobs=num_process)(delayed(deconv.deconvolution_DAPI)(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf) for i in tqdm(range(height_list.shape[0])) for j in range(width_list.shape[0]))


result_nd = np.zeros(imge_nd.shape,'float16')
for i in range(height_list.shape[0]):
    for j in range(width_list.shape[0]):
        tmp = results_subimg[i*width_list.shape[0]+j]
        if i==0 and j==0:
            result_nd[:,height_list[i,0]:height_list[i,1]-int(n_overlap/2),width_list[j,0]:width_list[j,1]-int(n_overlap/2)] = tmp[:,:tmp.shape[1]-int(n_overlap/2),:tmp.shape[2]-int(n_overlap/2)]
        elif i==0 and j==width_list.shape[0]-1:
            result_nd[:,height_list[i,0]:height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]] = np.maximum(result_nd[:,height_list[i,0]:height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]], tmp[:,:tmp.shape[1]-int(n_overlap/2),int(n_overlap/2):])
        elif i==height_list.shape[0]-1 and j==0:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]:width_list[j,1]-int(n_overlap/2)] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]:width_list[j,1]-int(n_overlap/2)], tmp[:,int(n_overlap/2):,:tmp.shape[2]-int(n_overlap/2)])
        elif i==height_list.shape[0]-1 and j==width_list.shape[0]-1:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]+int(n_overlap/2):width_list[j,1]] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]+int(n_overlap/2):width_list[j,1]], tmp[:,int(n_overlap/2):,int(n_overlap/2):])
        elif i==0:
            result_nd[:,height_list[i,0]:height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)] = np.maximum(result_nd[:,height_list[i,0]:height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)], tmp[:,:tmp.shape[1]-int(n_overlap/2),int(n_overlap/2):tmp.shape[2]-int(n_overlap/2)])
        elif j==0:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]:width_list[j,1]-int(n_overlap/2)] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]:width_list[j,1]-int(n_overlap/2)], tmp[:,int(n_overlap/2):tmp.shape[1]-int(n_overlap/2),:tmp.shape[2]-int(n_overlap/2)])
        elif i==height_list.shape[0]-1:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1],width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)], tmp[:,int(n_overlap/2):,int(n_overlap/2):tmp.shape[2]-int(n_overlap/2)])
        elif j==width_list.shape[0]-1:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]], tmp[:,int(n_overlap/2):tmp.shape[1]-int(n_overlap/2),int(n_overlap/2):])
        else:
            result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)] = np.maximum(result_nd[:,height_list[i,0]+int(n_overlap/2):height_list[i,1]-int(n_overlap/2),width_list[j,0]+int(n_overlap/2):width_list[j,1]-int(n_overlap/2)], tmp[:,int(n_overlap/2):tmp.shape[1]-int(n_overlap/2),int(n_overlap/2):tmp.shape[2]-int(n_overlap/2)])
        #result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]] = np.maximum(result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],results_subimg[i*width_list.shape[0]+j])


tifffile.imwrite(out_name+'-deconv.tif',np.round(result_nd).astype('uint16'), imagej=True)