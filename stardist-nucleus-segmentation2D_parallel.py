from stardist.models import StarDist2D
from csbdeep.utils import normalize
import sys
import numpy as np
from skimage import filters
from scipy import ndimage
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import tifffile

img_file_name = sys.argv[1] # 'old1.tif'
num_process = int(sys.argv[2]) # 2

if num_process>mp.cpu_count():
    num_process=mp.cpu_count()

output_pre = img_file_name.rsplit('.', 1)[0]
# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from skimage import io
imge_nd = tifffile.imread(img_file_name)
#labels, _ = model.predict_instances(normalize(img))

## allocate all subimages
#imge_nd = img.astype('float16')
#n_div=num_div
n_overlap=10 # number of pixel

(n_z,n_height,n_width) = imge_nd.shape
"""
height_list = np.zeros((n_div,2),'int')
width_list = np.zeros((n_div,2),'int')
for i in range(n_div):
    height_list[i,:]=[int(n_height/n_div*i-n_overlap),int(n_height/n_div*(i+1)+n_overlap)]
    width_list[i,:]=[int(n_width/n_div*i-n_overlap),int(n_width/n_div*(i+1)+n_overlap)]
"""
n_subimg_size=n_overlap*100
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

results_subimg_labels = Parallel(n_jobs=num_process)(delayed(model.predict_instances)(normalize(imge_nd[height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]])) for i in tqdm(range(height_list.shape[0])) for j in range(width_list.shape[0]))

result_nd = np.zeros(imge_nd.shape,'uint16')
max_label = 0
for i in range(height_list.shape[0]):
    for j in range(width_list.shape[0]):
        tmp = results_subimg_labels[i*width_list.shape[0]+j]
        tmp = tmp + max_label
        max_label = np.max(tmp)
        tmp_previous = result_nd[height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]]
        for idx in np.unique(tmp)



def shuffle_labels_notcontinuous(labels):
    random_label_dict = np.unique(labels)
    random_label_dict = random_label_dict[random_label_dict!=0]
    random_label_dict = dict(zip(random_label_dict, np.random.permutation(np.arange(1,len(random_label_dict)+1))))
    random_labels = np.zeros_like(labels)
    for i in random_label_dict:
        random_labels[labels==i] = random_label_dict[i]
    return random_labels

# filter low intensity region or saturated nucleus
img_raw_min,img_raw_max =img.min(), img.max()
img_sample_resample_uint8 = 255*1.0/(img_raw_max-img_raw_min) * (img-img_raw_min)
img_sample_resample_uint8 = img_sample_resample_uint8.astype(np.uint8)
val = filters.threshold_otsu(img_sample_resample_uint8)

cc_intensity = ndimage.mean(img_sample_resample_uint8, labels, range(0,np.max(labels)+1))
cc_mask = cc_intensity<1.0*val
labels[cc_mask[labels]] = 0

label_img_shuffle = shuffle_labels_notcontinuous(labels)

io.imsave(output_pre+'-stardist-label.tif',label_img_shuffle.astype('uint16'))