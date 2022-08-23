import numpy as np
from skimage import filters, restoration

def deconvolution_DAPI(image_ndarray,psf):
    """
    INPUT parameters:
    image_ndarray and psf: z,y,x ordered ndarray.
    Or y,x ordered ndarray.
    Output: z,y,x order or y,x ordered ndarray
    """
    deconvolved_RL = restoration.richardson_lucy(image_ndarray, psf, 20, clip=False)
    return deconvolved_RL

def psf_2D(emission_wavelength,pixelsize,NA,size=7):
    a = np.zeros((size, size))
    a[int((size-1)/2),int((size-1)/2)]=1
    return filters.gaussian(a,sigma=0.21*emission_wavelength/pixelsize/NA)

def deconvolution_wiener(image_ndarray,psf):
    deconvolved, _ = restoration.unsupervised_wiener(image_ndarray, psf, clip=False)
    return deconvolved

def func_divide_img_deconv(imge_nd,psf,n_div=3,n_overlap=0.00):
    """
    The input should be ndarray in z,y,x order.
    """
    (n_z,n_height,n_width) = imge_nd.shape
    height_list = np.zeros((n_div,2),'int')
    width_list = np.zeros((n_div,2),'int')
    for i in range(n_div):
        height_list[i,:]=[int(n_height/n_div*i-n_height/n_div*n_overlap),int(n_height/n_div*(i+1)+n_height/n_div*n_overlap)]
        width_list[i,:]=[int(n_width/n_div*i-n_width/n_div*n_overlap),int(n_width/n_div*(i+1)+n_width/n_div*n_overlap)]
    
    height_list[height_list<0]=0
    width_list[width_list<0]=0
    height_list[height_list>n_height-1]=n_height-1
    width_list[width_list>n_width-1]=n_width-1
    result_nd = np.zeros(imge_nd.shape,'float16')
    for i in range(n_div):
        for j in range(n_div):
            if i==0 and j==0:
                result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]] = deconvolution_DAPI(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf)
            elif i==0:
                tmp = deconvolution_DAPI(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf)
                result_nd[:,height_list[i,0]:height_list[i,1],width_list[j-1,1]:width_list[j,1]]=tmp[:,:,(-width_list[j,0]+width_list[j-1,1]):]
                result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j-1,1]] = np.maximum(result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j-1,1]],tmp[:,:,0:(width_list[j-1,1]-width_list[j,0])])
            elif j==0:
                tmp = deconvolution_DAPI(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf)
                result_nd[:,height_list[i-1,1]:height_list[i,1],width_list[j,0]:width_list[j,1]]=tmp[:,(-height_list[i,0]+height_list[i-1,1]):,:]
                result_nd[:,height_list[i,0]:height_list[i-1,1],width_list[j,0]:width_list[j,1]]=np.maximum(result_nd[:,height_list[i,0]:height_list[i-1,1],width_list[j,0]:width_list[j,1]],tmp[:,0:(height_list[i-1,1]-height_list[i,0]),:])
            else:
                tmp = deconvolution_DAPI(imge_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j,1]],psf)
                result_nd[:,height_list[i-1,1]:height_list[i,1],width_list[j-1,1]:width_list[j,1]]=tmp[:,(-height_list[i,0]+height_list[i-1,1]):,(-width_list[j,0]+width_list[j-1,1]):]
                result_nd[:,height_list[i,0]:height_list[i-1,1],width_list[j,0]:width_list[j,1]]=np.maximum(result_nd[:,height_list[i,0]:height_list[i-1,1],width_list[j,0]:width_list[j,1]],tmp[:,0:(height_list[i-1,1]-height_list[i,0]),:])
                result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j-1,1]] = np.maximum(result_nd[:,height_list[i,0]:height_list[i,1],width_list[j,0]:width_list[j-1,1]],tmp[:,:,0:(width_list[j-1,1]-width_list[j,0])])
    return result_nd