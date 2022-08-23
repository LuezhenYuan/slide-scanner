import openslide
from openslide import OpenSlideError
import PIL
from PIL import Image
from tifffile import TiffFile
import numpy as np
from skimage import restoration
def open_slide(filename):
  """
  Open a whole-slide image (*.svs, etc).
  Args:
    filename: Name of the slide file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filename)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide


def open_slide_tif_list(filename_list):
  """
  Open a whole-slide image (*.svs, etc).
  Args:
    filename: Name of the slide file.
  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  slide_list = []
  for i in range(len(filename_list)):
      try:
        slide = openslide.open_slide(filename_list[i])
      except OpenSlideError:
        slide = None
      except FileNotFoundError:
        slide = None
      slide_list.append(slide)
  return slide_list

def read_region_slide_tif_list(filename_list, location, level, size):
    """
    The input shoud be a pandas dataframe.
    It contains four columns: file_name,channel,z_stack,slide
    e.g.: file_channel4_zstack0.tif,4,0,OpenSlide('file_channel4_zstack0.tif...
    The fourth column is an openslide object
    Output: z,c,y,x order
    """
    channel_list = sorted([*{*filename_list['channel']}])
    z_stack_list = sorted([*{*filename_list['z_stack']}])
    region_nd = np.zeros(tuple([len(z_stack_list)])+tuple([len(channel_list)])+size)
    # slide_name_list[np.logical_and(slide_name_list['channel']==4,slide_name_list['z_stack']==0)]['file_name']
    for i_ch in range(len(channel_list)):
        for j_zstack in range(len(z_stack_list)):
            slide = filename_list[np.logical_and(filename_list['channel']==channel_list[i_ch],filename_list['z_stack']==z_stack_list[j_zstack])]['slide'].item()
            slide_region = slide.read_region(location,level,size)
            slide_region_nd = np.array(slide_region)
            region_nd[j_zstack,i_ch,:,:] = slide_region_nd[:,:,0] # seems like, only channel 0 is enough.
    return region_nd

def read_region_slide_tif_list_projection(filename_list, location, level, size):
    """
    The input shoud be a pandas dataframe.
    It contains four columns: file_name,channel,z_stack,slide
    e.g.: file_channel4_zstack0.tif,4,0,OpenSlide('file_channel4_zstack0.tif...
    The fourth column is an openslide object
    Output: c,y,x order
    """
    region_nd = read_region_slide_tif_list(filename_list, location, level, size)
    region_project = np.zeros((region_nd.shape[1],region_nd.shape[2],region_nd.shape[3]))
    for i in range(region_nd.shape[1]):
        region_project[i,:,:] = np.max(region_nd[:,i,:,:],axis=0)
    return region_project

    