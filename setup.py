import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TissueVisionPipeline",
    version="1.0.0",
    author="Nick Wang",
    author_email="nick.wang@mail.mcgill.ca",
    description="Automatic Image Stitching, Deep Learning based Cell Recognition, and Image Registration Pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mouse-Imaging-Centre/TissueVisionPipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=[
        'tools/TV_stitch.py',
        'tools/stacks_to_volume.py',
        'tools/MIP_first.py',
        'pipelines/TV_slice_recon.py',
        'pipelines/TV_minc_recon.py',
        'pipelines/TVBM.py'
    ],
    install_requires=[
        'fastCell',
        'pydpiper>=2.0.14',
    ]
)