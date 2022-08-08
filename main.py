"""Retinal Explant Area and Length Calculator

Produce area and length of tissues in a WSI. Done by identifying tissues at a 
low magnification and gathering their area and length statistics at a higher 
magnification. Saves area and length values as a JSON file.

Our pipeline queries a database for images at different resolutions and their 
respective information, as well as uploads masks to an image viewer. This script 
removes these processes, modifies, and assumes one inputs this information 
instead. (See line 73.)

Example of usage:
>>> python main.py "lomagimg.npy" himagimg.npy" "stats.json"

@Author: Kevin Marroquin, marroquk@gene.com
"""

## Imports
import numpy as np
import argparse
import json

# Scipy and scikit-image imports
from scipy import ndimage
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage.measure import regionprops, label
from skimage.morphology import (remove_small_holes, remove_small_objects, 
                                disk, square, thin)
from skimage.morphology import (binary_closing as closing, 
                                binary_opening as opening,
                                binary_dilation as dilation,
                                binary_erosion as erosion)

# For pruning in skeleton (FilFinder)
from fil_finder import FilFinder2D
from astropy import units as u

## Main code
def main(imglopath, imghipath, path2save, addmask=None, delmask=None):
    """Accepts two images of different resolutions and a path to save a JSON 
    file of results, being the areas and lengths of objects in question. 
    Optional addmask and delmask are available for manual inclusion/deletion of 
    tissues.
    
    Parameters
    ----------
    imglo : str
        Path to NxMxC RGB image to be analyzed at a low magnification.

    imghi : str
        Path to NxMxC RGB image to be analyzed at a high magnification.

    path2save : str
        Path to save JSON file.

    addmask : str, optional
        Path of NxM binary mask of manual additionals to be included in 
        analysis. To be accepted as the same low magnification as imglo.

    delmask : str, optional
        Path of NxM binary mask of manual deletions to be excluded from 
        analysis. To be accepted as the same low magnification as imglo.

    Returns
    -------
    None
        A JSON file of results is written in the given path2save.
    """

    # Variables and inital parameters
    lomag = 2.5
    himag = 20
    scale_lomag = 8 # Scale compared to original image resolution
    scale_himag =  1
    um_himag = 0.46 # Microns per pixel length at high mag
    
    # Reading images
    if imglopath.lower().endswith("npy"):
        imglo = np.load(imglopath)
    else:
        imglo = imread(imglopath)
      
    if imghipath.lower().endswith("npy"):
        imghi = np.load(imghipath)
    else:
        imghi = imread(imghipath)
        
    # Gathering manual annotations
    if addmask is None:
        addmask = np.zeros(imglo.shape[:2], dtype=bool)
    else:
        if addmask.lower().endswith("npy"):
            addmask = np.load(addmask)
        else:
            addmask = imread(addmask)
      
    if delmask is None:
        delmask = np.zeros(imglo.shape[:2], dtype=bool)
    else:
        if delmask.lower().endswith("npy"):
            delmask = np.load(delmask)
        else:
            delmask = imread(delmask)
    
    # Assertions for correctness
    assert (len(imglo.shape) == 3) & (imglo.shape[2] == 3), (
        "imglo not in NxMxC RGB format")
    assert (len(imghi.shape) == 3) & (imghi.shape[2] == 3), (
        "imghi not in NxMxC RGB format")
    assert imglo.shape[0] * scale_lomag == imghi.shape[0] * scale_himag, (
        "imglo N shape does not scale with imghi N")
    assert imglo.shape[1] * scale_lomag == imghi.shape[1] * scale_himag, (
        "imglo M shape does not scale with imghi M")
    assert imglo.shape[:2] == addmask.shape, (
        "imglo and addmask do not have the same NxM shape")
    assert imglo.shape[:2] == delmask.shape, (
        "imglo and delmask do not have the same NxM shape")
        
    # Creating masklomag and adding annotations
    masklomag = createMask(imglo, addmask, delmask)

    # Creating regions from a filled mask of masklowmag
    region_labels = label(masklomag)
    lomagregions = regionprops(region_labels)
    num_regions = len(lomagregions)
    
    # Iterating regions and calculating their initial statistics
    area_mask_vals = np.zeros(num_regions)
    length_mask_vals = np.zeros(num_regions)
    
    for region_num in range(num_regions):
        # Gather high mag image
        region_bbox = lomagregions[region_num].bbox
        bbox_scaled = [int(b * scale_lomag / scale_himag) for b in region_bbox]
        region_himag = imghi[bbox_scaled[0]: bbox_scaled[2], 
                             bbox_scaled[1]: bbox_scaled[3]]
        
        # Binary mask region of interest
        interest_region = (region_labels == (region_num + 1)
                          )[region_bbox[0]: region_bbox[2], 
                            region_bbox[1]: region_bbox[3]]
        interest_region = resize(interest_region, region_himag[:, :, 0].shape
                                ).astype(bool)
        
        # Creating area and length masks
        area_mask = createAreamask(region_himag, interest_region)
        length_mask = createLengthmask(region_himag, interest_region, himag)
        
        # Summing total pixels
        area_mask_vals[region_num] = np.sum(area_mask)
        length_mask_vals[region_num] = np.sum(length_mask)

    # Calculating stats and saving as a JSON file
    dict2save = {
        "Area, sq.um": list(area_mask_vals * um_himag**2),
        "Length, um": list(length_mask_vals * um_himag),
        "Number of Regions": region_num + 1
    }
    
    with open(path2save, 'w') as fp:
        json.dump(dict2save, fp,  indent=4)
        
    return

def createMask(img, addmask, delmask):
    """Creates a binary mask of an image's prominent regions given an add mask 
    and a del mask. 
    
    Parameters
    ----------
    img : NxMxC ndarray
        RGB image.

    addmask : NxM ndarray
        Binary mask of manual insertions with the same NxM shape as img. 
        
    delmask : NxM ndarray
        Binary mask of manual deletions with the same NxM shape as img. 

    Returns
    -------
    masklomag : ndarray
        Binary mask of prominent regions in img.
    """
    # Create binary mask
    masklomag = rgb2gray(img)
    masklomag = (gaussian(masklomag, sigma=3) * 255).astype(np.uint8)
    masklomag = masklomag < threshold_otsu(masklomag) #+ 25

    # Producing morphological functions on binary mask
    masklomag = closing(opening(masklomag, disk(1)), disk(1))
    masklomag = opening(closing(masklomag, disk(1)), disk(1))
    masklomag = closing(opening(masklomag, disk(2)), disk(2))
    masklomag = opening(closing(masklomag, disk(2)), disk(2))
    masklomag = dilation(masklomag, disk(5))
    masklomag = remove_small_objects(masklomag, 3000)
    masklomag = closing(opening(masklomag, disk(3)), disk(5))
    masklomag = remove_small_objects(masklomag, 5000)
    masklomag = dilation(masklomag, disk(9))
    masklomag = remove_small_holes(masklomag, 2000)
    masklomag = remove_small_objects(masklomag, 7000)

    # Add and removing maunal ROIs 
    masklomag = masklomag & ~delmask
    masklomag = remove_small_objects(masklomag, 500)
    masklomag = masklomag | addmask

    return masklomag
    
def createAreamask(img, interest_region):
    """Create area mask given an RGB img and an interest region mask.
    
    Parameters
    ----------
    img : NxMxC ndarray
        RGB image.

    interest_region : NxM ndarray
        Binary mask of prominent region in img with the same NxM shape as img. 

    Returns
    -------
    areamask : ndarray
        Binary mask of area in img.
    """
    # Creating binary mask
    graymask = rgb2gray(img)
    graymask_gauss = (gaussian(graymask, sigma=3) * 255).astype(np.uint8)
    himag_tissue = graymask_gauss < threshold_otsu(graymask_gauss) + 35 # Originally 25-35


    # Creating whitemask
    whitemask = graymask_gauss > min([threshold_otsu(graymask_gauss) + 40, 225]) # Originally 35-40
    whitemask = remove_small_objects(whitemask, 1000)

    whitemask = closing(whitemask, disk(1))
    whitemask = opening(whitemask, disk(1))
    whitemask = remove_small_objects(whitemask, 3000)

    whitemask = closing(whitemask, disk(2))
    whitemask = opening(whitemask, disk(2))
    whitemask = remove_small_objects(whitemask, 5000)

    # Combining whitemask and interest_region
    areamask = ~whitemask & interest_region
    areamask = remove_small_holes(areamask, 500)
    areamask = remove_small_objects(areamask, 500)
    areamask = closing(areamask, disk(1))
    areamask = opening(areamask, disk(1))

    areamask = remove_small_holes(areamask, 1000)
    areamask = closing(areamask, disk(2))
    areamask = opening(areamask, disk(2))
    areamask = remove_small_holes(areamask, 2000)
    areamask = areamask & interest_region

    return areamask
    
def createLengthmask(img, interest_region, mag):
    """Create length mask given an RGB img, an interest region, and a 
    magnification.
    
    Parameters
    ----------
    img : NxMxC ndarray
        RGB image.

    interest_region : NxM ndarray
        Binary mask of prominent region in img with the same NxM shape as img. 
        
    mag : num
        Magnification number to scale for FilFinder input.

    Returns
    -------
    lengthmask : ndarray
        Binary mask of length in img.
    """
    # Creating skeleton
    skeleton = thin(interest_region)
    skeleton = dilation(skeleton, square(2))
    
    # Pruning trees from skeleton
    fil = FilFinder2D(skeleton, distance=mag * u.millimeter, beamwidth=1 * u.pix,
                      mask=skeleton)

    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=250 * u.pix, skel_thresh=35 * u.pix, 
                          prune_criteria='length')
    
    lengthmask = fil.skeleton_longpath == 1
    
    # Creating whitemask
    whitemask = (rgb2gray(img) * 255).astype(np.uint8)
    whitemask = whitemask > min([threshold_otsu(whitemask) + 45, 240]) 
    whitemask = remove_small_objects(whitemask, 5000)
    whitemask = opening(closing(whitemask, disk(2)), disk(5))
    whitemask = opening(closing(whitemask, disk(3)), disk(8))
    whitemask = dilation(~whitemask, disk(10)) & interest_region
    whitemask = ndimage.binary_fill_holes(whitemask).astype(int)
    whitemask = remove_small_objects(whitemask, 5000)
    whitemask = ~dilation(whitemask, disk(13))
    
    lengthmask = lengthmask & ~whitemask & interest_region
    
    return lengthmask

## Inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imglopath', nargs='+',
                        help="Path to RGB image at a low mag scale")
    parser.add_argument('imghipath', nargs='+',
                        help="Path to RGB image at a high mag scale")
    parser.add_argument('path2save', nargs='+', 
                        type=str, help="Path to save results as a JSON file")
    parser.add_argument('--addmask', default=None, nargs='+',
                        help="Path to binary mask with imglo shape to add and analyze")
    parser.add_argument('--delmask', default=None, nargs='+',
                        help="Path to binary mask with imglo shape to delete from analysis")
    
    args = parser.parse_args()
    imglopath = ' '.join(args.imglopath)
    imghipath = ' '.join(args.imghipath)
    path2save = ' '.join(args.path2save)
    addmaskpath = ' '.join(args.addmask) if args.addmask is not None else None
    delmaskpath = ' '.join(args.delmask) if args.delmask is not None else None

    main(imglopath, imghipath, path2save, addmaskpath, delmaskpath)