#!/usr/bin/env python3
import os, re
from pathlib import Path; Path.ls = lambda x: list(x.iterdir())
from math import isnan
from typing import  Union, Dict

import pandas as pd

from pydpiper.core.stages import Stages, Result
from pydpiper.core.files import FileAtom
from pydpiper.execution.application import mk_application

from tissue_vision.reconstruction import TV_stitch_wrap
from tissue_vision.arguments import TV_stitch_parser

def find_mosaic_file(row) -> str:
    if len(sorted(Path(row.brain_directory).glob("Mosaic*txt"))) > 1:
        raise Exception("There are more than one Mosaic files found in %s" % row.brain_directory)
    return sorted(Path(row.brain_directory).glob("Mosaic*txt"))[0]

def read_mosaic_file(file: str) -> Dict:
    with open(file) as f: mosaic = f.read()
    keys = re.findall(r"(?<=\n).*?(?=:)", "\n" + mosaic)
    values = re.findall(r"(?<=:).*?(?=\n)", mosaic + "\n")
    return dict(zip(keys, values))

def tv_slice_recon_pipeline(options):
    output_dir = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    s = Stages()

    df = pd.read_csv(options.application.csv_file,
                     dtype={"brain_name":str, "brain_directory":str})
    # transforms = (mbm_result.xfms.assign(
    #     native_file=lambda df: df.rigid_xfm.apply(lambda x: x.source),

    df["mosaic_file"] = df.apply(lambda row: find_mosaic_file(row), axis = 1)
    df["mosaic_dictionary"] = df.apply(lambda row: read_mosaic_file(row.mosaic_file), axis = 1)
    df["number_of_slices"] = df.apply(lambda row: int(row.mosaic_dictionary["sections"]), axis = 1)
    df["interslice_distance"] = df.apply(lambda row: float(row.mosaic_dictionary["sectionres"])/1000, axis = 1)
    df["Zstart"] = df.apply(lambda row: 1 if isnan(row.Zstart) else row.Zstart, axis = 1)
    df["Zend"] = df.apply(lambda row: row.number_of_slices - row.Zstart + 1 if isnan(row.Zend) else row.Zend, axis=1)
    df["slice_directory"] = df.apply(lambda row: os.path.join(output_dir, pipeline_name + "_stitched", row.brain_name), axis=1)

#############################
# Step 1: Run TV_stitch.py
#############################
    #TODO surely theres a way around this?
    df = df.assign(TV_stitch_result = "")
    for index, row in df.iterrows():
        df.at[index,"TV_stitch_result"] = s.defer(TV_stitch_wrap(brain_directory = FileAtom(row.brain_directory),
                                                                 brain_name = row.brain_name,
                                                                 slice_directory = row.slice_directory,
                                                                 TV_stitch_options = options.TV_stitch,
                                                                 Zstart=row.Zstart,
                                                                 Zend=row.Zend,
                                                                 output_dir = output_dir
                                                                 ))
    df.drop(["mosaic_dictionary", "TV_stitch_result"], axis=1).to_csv("TV_brains.csv", index=False)
    df.explode("TV_stitch_result")\
        .assign(slice=lambda df: df.apply(lambda row: row.TV_stitch_result.path, axis=1))\
        .drop(["mosaic_dictionary", "TV_stitch_result"], axis=1)\
        .to_csv("TV_slices.csv", index=False)
    return Result(stages=s, output=())

tv_slice_recon_application = mk_application(parsers=[TV_stitch_parser],
                                      pipeline=tv_slice_recon_pipeline)

if __name__ == "__main__":
    tv_slice_recon_application()