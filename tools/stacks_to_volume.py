import os

from pyminc.volumes.factory import *
import numpy as np
import scipy.ndimage
import scipy.interpolate
import argparse

# taken from http://www.scipy.org/Cookbook/Rebinning
def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims.astype(np.int64))[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[tuple(cd)]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + list(range( ndims - 1))
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = list(range(np.rank(newcoords)))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None


if __name__ == "__main__":
    description = """
%(prog)s takes a series of two-dimensional images (tif, png, jpeg
 ... whatever can be read by python's PIL library) and converts them
 into a 3D MINC volume. Additional options control for resampling of
 slices - in the case of histology data, for example, the 2D slices
 might be at much higher resolution than the final desired 3D
 volume. In that case the slices are preblurred and then downsampled
 using nearest neighbour resampling before being inserted into the 3D
 volume.
"""
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_images', type=str, nargs="+",
                        help='')
    parser.add_argument('output_image', type=str,
                        help='')
    size = parser.add_argument_group("size")
    size.add_argument("--input-resolution", dest="input_resolution",
                          help="Input resolution in mm (i.e. the pixel size "
                          "assuming that it's isotropic) [default: %(default)s]",
                          type=float, default=0.00137)
    size.add_argument("--output-resolution", dest="output_resolution",
                          help="The desired output resolution in mm (i.e. "
                          "what the data will be resampled to) "
                          "[default: %(default)s]",
                          type=float, default=0.075)
    size.add_argument("--slice-gap", dest="slice_gap",
                          help="The slice gap in mm (i.e. the distance "
                          "between adjacent slices)[default: %(default)s]",
                          type=float, default=0.075)
    # add option for explicitly giving output matrix size

    preprocessing = parser.add_argument_group("preprocessing")
    preprocessing.add_argument("--scale-output", dest="scale_output",
                               help = "Scale the output by this value "
                                      "[default: %(default)s]",
                               type=float, default=1.0
                               )
    preprocessing.add_argument("--gaussian", action="store_const",
                                   help="Apply 2D gaussian (FWHM set based "
                                   "on input and output sizes)",
                                   const="gaussian", dest="preprocess",
                                   default=None)
    preprocessing.add_argument("--uniform", action="store_const",
                                   help="Apply 2D uniform filter (size based "
                                   "on input and output sizes)",
                                   const="uniform", dest="preprocess")
    preprocessing.add_argument("--uniform-sum", action="store_const",
                                   help="Apply 2D uniform filter and multiply "
                                   "it by filter volume to obtain a count. Use "
                                   "this option if, for example, the slices "
                                   "contain classified neurons.",
                                   const="uniform_sum", dest="preprocess")
    
    dim = parser.add_mutually_exclusive_group()
    dim.add_argument("--xyz", action="store_const",
                         help="XYZ dimension order",
                         const=("xspace", "yspace", "zspace"),
                         dest="dimorder",
                         default=("yspace", "zspace", "xspace"))
    dim.add_argument("--xzy", action="store_const",
                         help="XZY dimension order",
                         const=("xspace", "zspace", "yspace"),
                         dest="dimorder")
    dim.add_argument("--yxz", action="store_const",
                         help="YXZ dimension order",
                         const=("yspace", "xspace", "zspace"),
                         dest="dimorder")
    dim.add_argument("--yzx", action="store_const",
                         help="YZX dimension order [default]",
                         const=("yspace", "zspace", "xspace"),
                         dest="dimorder")
    dim.add_argument("--zxy", action="store_const",
                         help="ZXY dimension order",
                         const=("zspace", "xspace", "yspace"),
                         dest="dimorder")
    dim.add_argument("--zyx", action="store_const",
                         help="ZYX dimension order",
                         const=("zspace", "yspace", "xspace"),
                         dest="dimorder")

    args = parser.parse_args()
    
    # construct volume
    # need to know the number of slices
    n_slices = len(args.input_images)
    # need to know the size of the output slices - read in a single slice
    test_slice = scipy.ndimage.imread(args.input_images[0])
    slice_shape = np.array(test_slice.shape)
    size_fraction = args.input_resolution / args.output_resolution
    output_size = np.ceil(slice_shape * size_fraction).astype('int')
    filter_size = np.ceil(slice_shape[0] / output_size[0])

    vol = volumeFromDescription(args.output_image,
                                args.dimorder,
                                sizes=(n_slices,output_size[0],output_size[1]),
                                starts=(0,0,0),
                                steps=(args.slice_gap,
                                       args.output_resolution,
                                       args.output_resolution),
                                volumeType='ushort')
    for i in range(n_slices):
        print("In slice", i+1, "out of", n_slices)
        imslice = scipy.ndimage.imread(args.input_images[i])
        # normalize slice to lie between 0 and 1
        original_type_max = np.iinfo(imslice.dtype).max
        imslice = imslice.astype('float')
        imslice = imslice * (args.scale_output/original_type_max)

        # smooth the data depending on the chosen option
        if args.preprocess=="gaussian":
            imslice = scipy.ndimage.gaussian_filter(imslice, sigma=filter_size)
        if args.preprocess=="uniform" or args.preprocess=="uniform_sum":
            imslice = scipy.ndimage.uniform_filter(imslice, size=filter_size)
        if args.preprocess=="uniform_sum":
            imslice = imslice * filter_size * filter_size

        # downsample the slice
        o_imslice = congrid(imslice, output_size, 'neighbour')
        # add the downsampled slice to the volume
        vol.data[i,:,:] = o_imslice

    # finish: write the volume to file
    vol.writeFile()
    vol.closeVolume()
