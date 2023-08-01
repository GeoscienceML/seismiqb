import os
from contextlib import redirect_stderr
from nbtools import run_notebook
import argparse

parser = argparse.ArgumentParser(description='sgy converter')
parser.add_argument('-c', '--cube', help='cube name', default='*_*')
parser.add_argument('-f', '--force', action='store_true', help='recreate cube if exists')
parser.add_argument('-s', '--save', action='store_true', help='whether to save output notebook')
parser.add_argument('-p', '--postfix', action='store_true', help='add postfix to the created file')
parser.add_argument('--format', default='qsgy', help='format of cube')
parser.add_argument('--nonconvert', action='store_true', help='whether cubes should be converted')
parser.add_argument('--nonquantize', action='store_true', help='disable quantization of cube')

args = vars(parser.parse_args())

PATH = 'Convert_cubes.ipynb'
OUTPUT_PATH = 'Convert_cubes_executed.ipynb'

INPUTS = dict(
    CUBE = args['cube'],
    RECREATE = args['force'],
    CONVERT = not args['nonconvert'],
    FORMAT = args['format'],
    QUANTIZE = not args['nonquantize']
)

with open(os.devnull, 'w') as fnull:
    with redirect_stderr(fnull) as err:
        exceptions = run_notebook(PATH, INPUTS, inputs_pos=3, display_links=False, out_path_ipynb=OUTPUT_PATH)

if exceptions['failed cell number'] is not None:
    print('Cell:', exceptions['failed cell number'])
    print(exceptions['traceback'])

if not args['save']:
    os.remove(OUTPUT_PATH)
