"""
Utility to get primitives .json files
"""

import os
import json
import importlib
import inspect
import glob

ignore = ['PrimitiveBase', 'PrimitiveNotFittedError', 'UnsupervisedLearnerPrimitiveBase', 'SupervisedLearnerPrimitiveBase', 'TransformerPrimitiveBase', 'PrimitiveStep']
#skip_modukles = ['primitives.ts_forecasting.deep_ar.forecasting_deepar']

# List all the primitives
for p in glob.glob('primitives/*/*/*.py'):
    if p in glob.glob('primitives/*/utils/*') or p in glob.glob('primitives/*/*/__.init__.py'):
        continue
    f = p.replace('/', '.').replace('.py', '')
    lib = importlib.import_module(f)
    for l in dir(lib):
        if 'Primitive' in l and l not in ignore:
            pp = getattr(lib, l)
            print(f'Extracting {l}')
            md = pp.metadata.to_json_structure()
            name = md['python_path']
            os.chdir('/d3m-primitives/annotations')
            if not os.path.isdir(name):
                os.mkdir(name)
            os.chdir(name)
            v = getattr(lib, '__version__')
            if not os.path.isdir(v):
                os.mkdir(v)
            os.chdir(v)
            with open('primitive.json', 'w') as f:
                f.write(json.dumps(md, indent=4))
                f.write('\n')
