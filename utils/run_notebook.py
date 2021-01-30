import os
import argparse
import glob
import codecs

# +
import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

from nbparameterise import extract_parameters, replace_definitions
# -



# +
def compute_args(unknown):
    assert len(unknown) % 2 == 0
    
    params = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].replace('--', '')
        params[key] = unknown[i + 1]
    return params

def run_notebook_with_args(input_notebook, output_notebook, unknown):
    with codecs.open(input_notebook, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    orig_parameters = extract_parameters(nb)
            
    params = []
    args = compute_args(unknown)
    for k, v in args.items():
        for p in orig_parameters:
            if p.name == k:
                params.append(p.with_value(v))
                    
    if len(params) > 0:
        nb = replace_definitions(nb, params, execute=False)
        
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')    
    out = ep.preprocess(nb, {'metadata': {'path': '.'}})

    with codecs.open(output_notebook, encoding="utf-8", mode="w") as f:
        nbformat.write(nb, f)
# -



# +
parser = argparse.ArgumentParser()
parser.add_argument('--input_notebook', required=True)
parser.add_argument('--output_notebook', required=True)

args, unknown = parser.parse_known_args()

input_notebook = args.input_notebook
output_notebook = args.output_notebook
# -



run_notebook_with_args(input_notebook, output_notebook, unknown)
