from __future__ import print_function
from nd2reader import ND2Reader
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
from skimage import morphology, filters, measure, io
from scipy import ndimage

from skimage import io
img_file_name = sys.argv[1] # 'old1.tif'
img_0 = io.imread(img_file_name)
(img_pixel_microns) = (0.325) # Need to adjust based on imaging settings

output_file_Prefix = img_file_name.rsplit('.', 1)[0]
(img_width,img_height)=img_0.shape

arg_nucleus_diameter_micron = 15

nucleus_size_threshold = 0.5*np.pi*(arg_nucleus_diameter_micron/2)**2/img_pixel_microns/img_pixel_microns


#img_0_maxz = np.amax(img_0, axis=0)


local_Otsu_radius = int(arg_nucleus_diameter_micron*1.0/img_pixel_microns/2 + 0.5)

# Convert image to uint8 data type
img_raw_min,img_raw_max =img_0.min(), img_0.max()
img_sample_resample_uint8 = 255*1.0/(img_raw_max-img_raw_min) * (img_0-img_raw_min)
img_sample_resample_uint8 = img_sample_resample_uint8.astype(np.uint8)

del img_0

img_sample_resample_uint8_median = ndimage.median_filter(img_sample_resample_uint8, size=1)
#img_sample_resample_uint8_gaussian = ndimage.gaussian_filter(img_sample_resample_uint8_median, sigma=1)
del img_sample_resample_uint8

## Local Otsu, for 2D
mask_disk = morphology.disk(local_Otsu_radius).astype(np.bool_)

img_sample_resample_uint8_local_otsu_slice = filters.rank.otsu(img_sample_resample_uint8_median,mask_disk)
local_otsu_binarized_img = img_sample_resample_uint8_median >= 0.8*img_sample_resample_uint8_local_otsu_slice

## Rough binarized image
val = filters.threshold_otsu(img_sample_resample_uint8_median)

rough_binarized_img = img_sample_resample_uint8_median >= 0.8*val
rough_binarized_img_maximumfilter = morphology.binary_dilation(rough_binarized_img,morphology.disk(int(local_Otsu_radius/3),dtype=np.uint8)).astype(np.bool_)
binarized_img_improved = np.logical_and(rough_binarized_img_maximumfilter, local_otsu_binarized_img)
binarized_img_improved_fill = ndimage.morphology.binary_fill_holes(binarized_img_improved).astype(np.bool_)
'''
# watershed

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

distance = ndimage.distance_transform_edt(binarized_img_improved_fill)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((int(local_Otsu_radius/0.5),int(local_Otsu_radius/0.5))), labels=binarized_img_improved_fill)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=binarized_img_improved_fill)

# find connected component and size.
CC = ndimage.find_objects(labels_ws)
'''
labeled_img_0, num_label_0 = ndimage.label(binarized_img_improved_fill)
cc_area = ndimage.sum(binarized_img_improved_fill, labeled_img_0, range(0,np.max(labeled_img_0)+1)) # label to the values of the array, sum of the values for certain labels.
label_img=np.copy(labeled_img_0)
volume_mask = (cc_area<np.pi*(5.0/img_pixel_microns)**2/5) | (cc_area>np.pi*(5.0/img_pixel_microns)**2*5)
label_img[volume_mask[labeled_img_0]] = 0

cc_intensity = ndimage.mean(img_sample_resample_uint8_median, labeled_img_0, range(0,np.max(labeled_img_0)+1))
intensity_mask = cc_intensity<1.0*val
label_img[intensity_mask[labeled_img_0]] = 0

saturated_area = ndimage.sum(img_sample_resample_uint8_median==255, labeled_img_0, range(0,np.max(labeled_img_0)+1))
saturation_mask = 1.0*saturated_area/cc_area >0.1
label_img[saturation_mask[labeled_img_0]] = 0

def shuffle_labels_notcontinuous(labels):
    random_label_dict = np.unique(labels)
    random_label_dict = random_label_dict[random_label_dict!=0]
    random_label_dict = dict(zip(random_label_dict, np.random.permutation(np.arange(1,len(random_label_dict)+1))))
    random_labels = np.zeros_like(labels)
    for i in random_label_dict:
        random_labels[labels==i] = random_label_dict[i]
    return random_labels

label_img_shuffle = shuffle_labels_notcontinuous(label_img)

## save label images
io.imsave(output_file_Prefix+'-int-label.tif',label_img_shuffle.astype('uint16'))
#io.imsave(output_file_Prefix+'_img_2D.tif',img_sample_resample_uint8.astype('uint8'))

print(output_file_Prefix+' labelling done!')
# python 2D_segment_save_tif.py old1.tif old1-DAPI