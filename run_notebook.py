#!python

import os

import papermill as pm

print(os.getcwd())

EXPORT_NAME = 'training-exports.zip'
EXPORT_LOCATION = '../exports/en-tw-transformer-nmt/'
PROJECT_DIR = 'projects/en-tw-transformer-nmt'
TRAIN_SAMPLE = 100

pm.execute_notebook(
    f'{PROJECT_DIR}/main.ipynb',
    f'{PROJECT_DIR}/out.ipynb',
    cwd=os.getcwd(),
    parameters=dict(exports_dir=EXPORT_LOCATION, training_sample=TRAIN_SAMPLE, num_attention_layers=4)
)
