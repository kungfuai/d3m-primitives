"""
Utility to generate primitives .json annotations
"""

import os
import json
import importlib
import glob

ignore = [
    'PrimitiveBase', 
    'PrimitiveNotFittedError', 
    'UnsupervisedLearnerPrimitiveBase', 
    'SupervisedLearnerPrimitiveBase', 
    'TransformerPrimitiveBase', 
    'PrimitiveStep'
]

# List all the primitives
for p in glob.glob('kf_d3m_primitives/*/*/*.py'):
    if p in glob.glob('kf_d3m_primitives/*/utils/*') or p in glob.glob('kf_d3m_primitives/*/*/__.init__.py'):
        continue
    f = p.replace('/', '.').replace('.py', '')
    module = importlib.import_module(f)
    for c in dir(module):
        if 'Primitive' in c and c not in ignore:
            primitive = getattr(module, c)
            primitive_json = primitive.metadata.to_json_structure()
            primitive_name = primitive_json['python_path']
            os.chdir('/annotations')
            if not os.path.isdir(primitive_name):
                os.mkdir(primitive_name)
            os.chdir(primitive_name)
            version = getattr(module, '__version__')
            if not os.path.isdir(version):
                os.mkdir(version)
            os.chdir(version)
            with open('primitive.json', 'w') as f:
                f.write(json.dumps(primitive_json, indent=4))
                f.write('\n')
            print(f'Generated json annotation for {c}')
