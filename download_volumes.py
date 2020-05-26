"""
Utility to download large static files needed for some primitives
"""

import os
import subprocess

large_file_primitives = [
    "d3m.primitives.data_cleaning.column_type_profiler.Simon",
    "d3m.primitives.data_cleaning.geocoding.Goat_forward",
    "d3m.primitives.data_cleaning.text_summarization.Duke",
    "d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec",
    "d3m.primitives.object_detection.retinanet.ObjectDetectionRN",
    "d3m.primitives.classification.inceptionV3_image_feature.Gator",
]

for large_file_primitive in large_file_primitives:
    subprocess.run(
        [
            "python3",
            "-m",
            "d3m",
            "primitive",
            "download",
            "-p",
            large_file_primitive,
            "-o",
            "/static_volumes"
        ],
        check = True
    )
    print(f'Downloaded large static file for primitive: {large_file_primitive}')
