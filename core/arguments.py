from configargparse import ArgParser, Namespace
from pydpiper.core.util import NamedTuple
from pydpiper.core.arguments import BaseParser, AnnotatedParser

def _mk_TV_stitch_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--scale-output", dest="scale_output",
                   type=int,
                   default=None,  # TODO raise a warning when this isn't specified
                   help="Multiply slice images by this value before saving to file")
    p.add_argument("--keep-stitch-tmp", dest="keep_tmp",
                   action="store_true", default=False,
                   help="Keep temporary files from TV_stitch.")
    return p
TV_stitch_parser = AnnotatedParser(parser=BaseParser(_mk_TV_stitch_parser(), "TV_stitch"),
                                   namespace="TV_stitch")

def _mk_deep_segment_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--temp-dir", dest="temp_dir", type=str, default=None)
    p.add_argument("--deep-segment-pipeline", dest="deep_segment_pipeline",
                   type=str,
                   default=None,
                   help="This is the segmentation pipeline")
    p.add_argument("--anatomical-name", dest="anatomical_name",
                   type=str,
                   default="cropped",
                   help="Specify the name of the anatomical images outputted by deep_segment.py")
    p.add_argument("--count-name", dest="count_name",
                   type=str,
                   default="count",
                   help="Specify the name of the count images outputted by deep_segment.py")
    p.add_argument("--segment-name", dest="segment_name",
                   type=str,
                   default=None,
                   help="Specify the name of the segmentation images outputted by deep_segment.py")
    p.add_argument("--outline-name", dest="outline_name",
                   type=str,
                   default=None,
                   help="Specify the name of the outline images outputted by deep_segment.py")
    p.add_argument("--qc-fraction", dest="qc_fraction",
                   type=float,
                   default=0.1,
                   help="""
                   Specify what fraction [0, 1.0] of slices should have outline images output (equally spaced).
                   This is only used if the qc column does not exist in csv-file.
                   """)
    p.add_argument("--cell-min-area", dest="cell_min_area",
                   type=int,
                   default=None,
                   help="""
                   The neural network may mistakenly segment stray pixels as cells.
                   All segmented cells with area less than the value specified by --cell-min-area will be ignored.
                   """)
    p.add_argument("--cell-mean-area", dest="cell_mean_area",
                   type=float,
                   default=None,
                   help="""
                   Individual cells that aren't touching are processed by being reduced to a single point at their centroid.
                   Clusters of cells area identified by the maximum area criterium specified by --cell-max-area.
                   The number of cells contained in the cluster is found by dividing the cluster's total area
                   by the mean area criterium specified by --cell-mean-area. That many cell centroids are
                   randomly and uniformly sampled inside the cluster.
                   Note that this will not work out of the box. Your neural network provided to --deep-segment-pipeline
                   must be trained to recognize clusters of cells in addition to individual cells.
                    """)
    p.add_argument("--cell-max-area", dest="cell_max_area",
                   type=int,
                   default=None,
                   help="See help for --cell-mean-area")
    return p
deep_segment_parser = AnnotatedParser(parser=BaseParser(_mk_deep_segment_parser(), "deep_segment"),
                                      namespace="deep_segment")

def _mk_stacks_to_volume_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--input-resolution", dest="input_resolution",
                   type=float,
                   default=0.00137,
                   help="The raw in-plane resolution of the tiles in mm. [default = %(default)s]")
    p.add_argument("--plane-resolution", dest="plane_resolution",
                   type=float,
                   default=None,
                   help="The output in-plane resolution of the tiles in mm")
    p.add_argument("--manual-scale-output", dest="manual_scale_output",
                   action="store_true", default=False,
                   help="The purpose of this option is to correct for when brains have been imaged using different "
                        "interslice distances."
                        "If true [default = %(default)s], your input to --csv-file must have a scale_output column. "
                        "The stacked count MINC file will have its values scaled by that number. "
                        "If false, each brain's count slices will be scaled by its interslice distance divided by the "
                        "the minimum interslice distance of all brains. Each brain's scalar value will be reflected "
                        "in the output csv files."
                   )
    return p
stacks_to_volume_parser = AnnotatedParser(parser=BaseParser(_mk_stacks_to_volume_parser(), "stacks_to_volume"),
                                      namespace="stacks_to_volume")

def _mk_autocrop_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--x-pad", dest="x_pad",
                   type=str,
                   default='0,0',
                   help="Padding in mm will be added to each sides. [default = %(default)s]")
    p.add_argument("--y-pad", dest="y_pad",
                   type=str,
                   default='0,0',
                   help="Padding in mm will be added to each sides. [default = %(default)s]")
    p.add_argument("--z-pad", dest="z_pad",
                   type=str,
                   default='0,0',
                   help="Padding in mm will be added to each side. [default = %(default)s]")
    return p
autocrop_parser = AnnotatedParser(parser=BaseParser(_mk_autocrop_parser(), "autocrop"), namespace="autocrop")

def _mk_consensus_to_atlas_parser():
    p = ArgParser(add_help=False)
    p.add_argument("--atlas-target", dest="atlas_target",
                   type=str,
                   default=None,
                   help="Register the consensus average to the ABI Atlas")
    p.add_argument("--atlas-target-label", dest="atlas_target_label",
                   type=str,
                   default=None,
                   help="Register the consensus average to the ABI Atlas")
    p.add_argument("--atlas-target-mask", dest="atlas_target_mask",
                   type=str,
                   default=None,
                   help="Register the consensus average to the ABI Atlas")
    return p
consensus_to_atlas_parser = AnnotatedParser(parser=BaseParser(_mk_consensus_to_atlas_parser(), 'consensus_to_atlas'),
                                            namespace='consensus_to_atlas')