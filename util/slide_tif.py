from tifffile import TiffFile
import numpy as np
def get_crop(page, i0, j0, h, w):
    """Extract a crop from a TIFF image file directory (IFD).
    
    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit into RAM.
    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    i0, j0: int
        Coordinates of the top left corner of the desired crop.
    h: int
        Desired crop height.
    w: int
        Desired crop width.
    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.
    """

    if not page.is_tiled:
        raise ValueError("Input page must be tiled.")

    im_width = page.imagewidth
    im_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
        raise ValueError("Requested crop area is out of image bounds.")

    tile_width, tile_height = page.tilewidth, page.tilelength
    i1, j1 = i0 + h, j0 + w

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty((page.imagedepth,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    page.samplesperpixel), dtype=page.dtype)

    fh = page.parent.filehandle

    jpegtables = page.tags.get('JPEGTables', None)
    if jpegtables is not None:
        jpegtables = jpegtables.value

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    return out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]

def read_region_tif_list_allchannel(filename_list, location, size):
    """
    The input shoud be a pandas dataframe.
    It contains three columns: file_name,channel,z_stack
    e.g.: file_channel4_zstack0.tif,4,0
    Output: z,c,y,x order
    """
    channel_list = sorted([*{*filename_list['channel']}])
    z_stack_list = sorted([*{*filename_list['z_stack']}])
    region_nd = np.zeros(tuple([len(z_stack_list)])+tuple([len(channel_list)])+size)
    for i_ch in range(len(channel_list)):
        for j_zstack in range(len(z_stack_list)):
            slide_name = filename_list[np.logical_and(filename_list['channel']==channel_list[i_ch],filename_list['z_stack']==z_stack_list[j_zstack])]['file_name'].values[0]
            m_tif = TiffFile(slide_name)
            m_tif_page = m_tif.pages[0]
            m_tif_region = get_crop(m_tif_page,*location,*size)
            region_nd[j_zstack,i_ch,:,:] = m_tif_region[0,:,:,0]
    return region_nd

def read_region_tif_list_onechannel(filename_list, location, size, channel):
    """
    The input shoud be a pandas dataframe.
    It contains three columns: file_name,channel,z_stack
    e.g.: file_channel4_zstack0.tif,4,0
    Output: z,y,x order
    """
    z_stack_list = sorted([*{*filename_list['z_stack']}])
    region_nd = np.zeros(tuple([len(z_stack_list)])+size)
    for j_zstack in range(len(z_stack_list)):
        slide_name = filename_list[np.logical_and(filename_list['channel']==channel,filename_list['z_stack']==z_stack_list[j_zstack])]['file_name'].values[0]
        m_tif = TiffFile(slide_name)
        m_tif_page = m_tif.pages[0]
        m_tif_region = get_crop(m_tif_page,*location,*size)
        region_nd[j_zstack,:,:] = m_tif_region[0,:,:,0]
    return region_nd

def read_full_tif_list_onechannel(filename_list, channel):
    """
    The input shoud be a pandas dataframe.
    It contains three columns: file_name,channel,z_stack
    e.g.: file_channel4_zstack0.tif,4,0
    Output: z,y,x order
    """
    z_stack_list = sorted([*{*filename_list['z_stack']}])
    slide_name = filename_list.iloc[0]['file_name']
    m_tif = TiffFile(slide_name)
    m_tif_page = m_tif.pages[0]
    size = (m_tif_page.imagelength-1,m_tif_page.imagewidth-1)
    region_nd = np.zeros(tuple([len(z_stack_list)])+size, dtype='uint8')
    for j_zstack in range(len(z_stack_list)):
        slide_name = filename_list[np.logical_and(filename_list['channel']==channel,filename_list['z_stack']==z_stack_list[j_zstack])]['file_name'].values[0]
        m_tif = TiffFile(slide_name)
        m_tif_page = m_tif.pages[0]
        m_tif_region = get_crop(m_tif_page,0,0,*size)
        region_nd[j_zstack,:,:] = m_tif_region[0,:,:,0]
    return region_nd

def z_projection_max(imge_nd):
    """
    The input should be ndarray in z,y,x order.
    The outpt is y,x order.
    """
    return np.amax(imge_nd,axis=0)
