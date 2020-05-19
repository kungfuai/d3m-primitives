"""
Utility to get generate all submission pipelines for all primitives
"""

import os
from importlib import import_module
from glob import glob

ignore = [
    'Pipeline', 
    'PipelineBase',
]

prims_to_pipelines = {

}
pipelines_to_datasets = {

}

args:
    -gpu, -cpu: T/F/None
    -need-distil, -no-distil: T/F/None

for primitive_type in glob('/kf-d3m-primitives/primitives/*/*/'):
    if primitive_type in glob('/kf-d3m-primitives/primitives/*/utils') or primitive_type in glob('/kf-d3m-primitives/primitives/*/__.init__.py'):
        continue
    m = glob(primitive_type + '/*_pipeline.py')
    module = import_module(m)
    for c in dir(module)
        if 'Pipeline' in c and c not in ignore:
            # change to correct dir
            # mk pipelines and pipeline runs folder
            # for pipeline: 
                # write pipeline
                # for pipeline_run in pipeline:
                    # write pipeline_run
