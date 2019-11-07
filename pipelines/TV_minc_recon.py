#!/usr/bin/env python3
import os

import pandas as pd

from pydpiper.core.stages import Stages, Result
from pydpiper.execution.application import mk_application
from pydpiper.core.files import FileAtom
from pydpiper.minc.files import MincAtom
from pydpiper.minc.registration import autocrop, create_quality_control_images

from tissue_vision.arguments import deep_segment_parser, stacks_to_volume_parser, autocrop_parser
from tissue_vision.reconstruction import deep_segment, stacks_to_volume

def tv_recon_pipeline(options):
    output_dir = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    s = Stages()

    slices_df = pd.read_csv(options.application.csv_file,
                     dtype={"brain_name": str, "brain_directory": str, "slice_directory": str})

    if "qc" not in slices_df.columns:
        slices_df["qc"] = False
        step = int(1 / options.deep_segment.qc_fraction)
        slices_df.qc[0::step] = True

#############################
# Step 1: Run deep_segment.py
#############################
    #TODO surely theres a way around deep_segment_result=""?
    slices_df = slices_df.assign(
        deep_segment_result="",
        segmentation_directory = lambda df: df.apply(
            lambda row: os.path.join(output_dir, pipeline_name + "_deep_segmentation", row.brain_name), axis=1)
    )
    for index, row in slices_df.iterrows():
        slices_df.at[index,"deep_segment_result"] = s.defer(deep_segment(image = FileAtom(row.slice,
                                                                                          output_sub_dir = row.segmentation_directory),
                                                                         deep_segment_pipeline = FileAtom(options.deep_segment.deep_segment_pipeline),
                                                                         anatomical_suffix = options.deep_segment.anatomical_name,
                                                                         count_suffix = options.deep_segment.count_name,
                                                                         outline_suffix = options.deep_segment.outline_name if row.qc else None,
                                                                         cell_min_area = options.deep_segment.cell_min_area,
                                                                         cell_mean_area = options.deep_segment.cell_mean_area,
                                                                         cell_max_area = options.deep_segment.cell_max_area,
                                                                         temp_dir = options.deep_segment.temp_dir
                                                                         ))
        #hacky solution requires deep_segment() returns in that order
        #https://stackoverflow.com/questions/35491274/pandas-split-column-of-lists-into-multiple-columns
        slices_df[["anatomical_result", "count_result", "outline_result"]] = \
            pd.DataFrame(slices_df.deep_segment_result.values.tolist())

#############################
# Step 2: Run stacks_to_volume.py
#############################
    #This is annoying... If I add anything to slices_df, I will have to delete it here as well
    mincs_df = slices_df.drop(['slice', 'deep_segment_result', "anatomical_result", "count_result", "outline_result", "qc"], axis=1) \
        .drop_duplicates().reset_index(drop=True)\
        .assign(
        anatomical_list = slices_df.groupby("brain_name")['anatomical_result'].apply(list).reset_index(drop=True),
        count_list=slices_df.groupby("brain_name")['count_result'].apply(list).reset_index(drop=True),
        #the above is so hacky...
        stacked_directory=lambda df: df.apply(
            lambda row: os.path.join(output_dir, pipeline_name + "_stacked", row.brain_name), axis=1),
    )
    mincs_df = mincs_df.assign(
        anatomical_stacked_MincAtom=lambda df: df.apply(
            lambda row: MincAtom(
                os.path.join(row.stacked_directory,
                             row.brain_name + "_" + options.deep_segment.anatomical_name + "_stacked.mnc")
            ), axis=1
        ),
        count_stacked_MincAtom=lambda df: df.apply(
            lambda row: MincAtom(
                os.path.join(row.stacked_directory,
                             row.brain_name + "_" + options.deep_segment.count_name + "_stacked.mnc")
            ), axis=1
        )
    )
    if not options.stacks_to_volume.manual_scale_output:
        mincs_df["scale_output"] = mincs_df.interslice_distance/mincs_df.interslice_distance.min()

    for index, row in mincs_df.iterrows():
        s.defer(stacks_to_volume(
            slices = row.anatomical_list,
            output_volume = row.anatomical_stacked_MincAtom,
            z_resolution = row.interslice_distance,
            stacks_to_volume_options=options.stacks_to_volume,
            uniform_sum=False
        ))
        s.defer(stacks_to_volume(
            slices=row.count_list,
            output_volume=row.count_stacked_MincAtom,
            z_resolution=row.interslice_distance,
            stacks_to_volume_options=options.stacks_to_volume,
            scale_output = row.scale_output,
            uniform_sum=True
        ))
#############################
# Step 3: Run autocrop to resample to isotropic
#############################
    for index, row in mincs_df.iterrows():
        mincs_df.at[index,"anatomical_isotropic_result"] = s.defer(autocrop(
            img = row.anatomical_stacked_MincAtom,
            isostep = options.stacks_to_volume.plane_resolution,
            suffix = "isotropic"
        ))
        mincs_df.at[index, "count_isotropic_result"] = s.defer(autocrop(
            img=row.count_stacked_MincAtom,
            isostep=options.stacks_to_volume.plane_resolution,
            suffix="isotropic",
            nearest_neighbour = True
        ))

#############################
    slices_df = slices_df.assign(
        anatomical_slice = lambda df: df.apply(lambda row: row.anatomical_result.path, axis=1),
        count_slice=lambda df: df.apply(lambda row: row.count_result.path, axis=1),
        outline_slice=lambda df: df.apply(lambda row: row.outline_result.path if row.outline_result else None, axis=1),
    )
    slices_df.drop(slices_df.filter(regex='.*_directory.*|.*_result.*'), axis=1)\
        .to_csv("TV_processed_slices.csv", index=False)

    mincs_df = mincs_df.assign(
        anatomical=lambda df: df.apply(lambda row: row.anatomical_isotropic_result.path, axis=1),
        count=lambda df: df.apply(lambda row: row.count_isotropic_result.path, axis=1),
    )
    mincs_df.drop(mincs_df.filter(regex='.*_result.*|.*_list.*|.*_MincAtom.*'), axis=1)\
        .to_csv("TV_mincs.csv", index=False)
    #TODO overlay them
    # s.defer(create_quality_control_images(imgs=reconstructed_mincs, montage_dir = output_dir,
    #     montage_output=os.path.join(output_dir, pipeline_name + "_stacked", "reconstructed_montage"),
    #                                       message="reconstructed_mincs"))

    #TODO
    # s.defer(create_quality_control_images(imgs=all_anatomical_pad_results, montage_dir=output_dir,
    #                                       montage_output=os.path.join(output_dir, pipeline_name + "_stacked",
    #                                                                   "%s_montage" % anatomical),
    #                                       message="%s_mincs" % anatomical))
    # s.defer(create_quality_control_images(imgs=all_count_pad_results, montage_dir=output_dir,
    #                                       montage_output=os.path.join(output_dir, pipeline_name + "_stacked",
    #                                                                   "%s_montage" % count),
    #                                       auto_range=True,
    #                                       message="%s_mincs" % count))
    return Result(stages=s, output=())

tv_recon_application = mk_application(parsers = [deep_segment_parser,
                                                 stacks_to_volume_parser,
                                                 autocrop_parser],
                                      pipeline = tv_recon_pipeline)

if __name__ == "__main__":
    tv_recon_application()