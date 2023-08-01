""" Script for running tests notebooks with provided parameters.

Each test execution is controlled by the following constants that are declared in the `common_params` dict:

REMOVE_EXTRA_FILES : bool
    Whether to remove extra files such as executed notebooks without failures.
SHOW_FIGURES : bool
    Whether to show additional figures in the executed notebooks.
    Showing some figures can be useful for finding out the reason for the tests failure.
VERBOSE : bool
    Whether to print in the terminal additional information from tests.

Other noteworthy variables for tests control are:

TESTS_ROOT_DIR : str
    Path to the directory for saving results and temporary files for all tests
    (executed notebooks, logs, data files like cubes, etc.).
    Note that the directory will be removed if `REMOVE_ROOT_DIR` is True and no one test failed.
REMOVE_ROOT_DIR : bool
    Whether to remove `TESTS_ROOT_DIR` after execution in case of all tests completion without failures.

Another important script part is the `notebooks_params` variable which manages notebooks execution order,
internal parameter values and outputs variables names for each individual test.
To add a new test case you just need to add a configuration tuple (notebook_path, params_dict) in it, where
the `params_dict` may have optional keys 'inputs' and 'outputs':
    - 'inputs' is a dict with test parameters to pass to the test notebook execution,
    - 'outputs' contains names of variables to return from the test notebook.

After all parameters initializations the `test_run_notebook` function is called.
Under the hood, the function parses test arguments, runs test notebooks with given configurations,
catches execution information such as traceback and internal variables values, and provides them to the terminal output.

Output file names processed from the execution count, the executed notebook name and passed inputs into it.
Correspondence between out file name and its test configuration is saved in
`seismiqb/tests/tests_root_dir_*/out_files_info.json`.
"""
import os
import json
import re
import shutil
import subprocess
import tempfile
import pytest
from nbtools import run_notebook


# Base tests variables for entire test process
pytest.failed = False
pytest.out_files_info = {}
BASE_DIR =  os.path.normpath(os.getenv('BASE_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../seismiqb')))
TESTS_DIR = os.path.join(BASE_DIR, 'tests')
git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
pytest.TESTS_ROOT_DIR = os.getenv('SEISMIQB_TESTS_ROOT_DIR', None)
REMOVE_ROOT_DIR = bool(int(os.getenv('SEISMIQB_TESTS_REMOVE_ROOT_DIR', '1')))

# Parameters for each test notebooks
common_params = {
    'REMOVE_EXTRA_FILES': bool(int(os.getenv('SEISMIQB_TESTS_REMOVE_EXTRA_FILES', '1'))),
    'SHOW_FIGURES': bool(int(os.getenv('SEISMIQB_TESTS_SHOW_FIGURES', '0'))),
    'VERBOSE': bool(int(os.getenv('SEISMIQB_TESTS_VERBOSE', '1')))
}

TESTS_NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'tests/notebooks/') # path to the directory with tests notebooks
# TUTORIALS_DIR = os.path.join(BASE_DIR, 'tutorials/')             # path to the directory with tutorials

geometry_formats = ['sgy', 'qsgy', 'hdf5', 'qhdf5']
notebooks_params = (
    # Tests configurations:
    # (notebook_path, {'inputs': dict (optional), 'outputs': str or list of str (optional)})

    # CharismaMixin test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'charisma_test.ipynb'), {}),

    # Geometry tests
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_01_preparation.ipynb'),
     {'inputs': {'FORMATS': geometry_formats}}),

    *[(os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_02_data_format.ipynb'),
       {'inputs': {'FORMAT': data_format}, 'outputs': 'timings'}) for data_format in geometry_formats],

    (os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_03_transforms.ipynb'),
     {'inputs': {'FORMATS': 'sgy'}}),

    # Horizon tests
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_01_preparation.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_02_base.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_03_attributes.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_04_processing.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_05_extraction.ipynb'), {}),

    # Fault tests
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'fault_test_01_preparation.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'fault_test_02_base.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'fault_test_03_sticks_processing.ipynb'), {}),
    # (os.path.join(TESTS_NOTEBOOKS_DIR, 'fault_test_04_mask_creation.ipynb'), {}), # TODO: re-enable after updating BF

    # Cache test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'cache_test.ipynb'), {}),

    # TODO: add tutorials
    # (os.path.join(TUTORIALS_DIR, '01_Geometry_part_1.ipynb'), {})
)


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys, finalize_fixture):
    """ Run tests notebooks using kwargs and print outputs in the terminal. """
    # Parse kwargs
    pytest.TESTS_ROOT_DIR = pytest.TESTS_ROOT_DIR or tempfile.mkdtemp(prefix=f'tests_root_dir_{git_hash}_', dir=TESTS_DIR)

    path_ipynb, params = notebook_kwargs
    filename = os.path.basename(path_ipynb)

    outputs = params.pop('outputs', None)
    inputs = params.pop('inputs', {})
    inputs_repr = str(inputs) # for printing output info
    out_filename = create_output_filename(input_filename=filename, inputs=inputs)
    pytest.out_files_info[out_filename] = {'filename': filename, 'inputs': inputs.copy()}

    inputs.update(common_params)

    # Run test notebook
    out_path_ipynb = os.path.join(pytest.TESTS_ROOT_DIR, out_filename)
    exec_res = run_notebook(path=path_ipynb, inputs=inputs, outputs=outputs,
                            inputs_pos=2, working_dir=pytest.TESTS_ROOT_DIR,
                            out_path_ipynb=out_path_ipynb, display_links=False)

    if not exec_res['failed'] and common_params['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)
        del pytest.out_files_info[out_filename]

    pytest.failed = pytest.failed or exec_res['failed']

    # Terminal output
    with capsys.disabled():
        notebook_info = f"`{filename}`{' with inputs=' + inputs_repr if inputs_repr!='{}' else ''}"

        # Extract traceback
        if exec_res['failed']:
            print(exec_res.get('traceback', ''))
            print(f"\n{notebook_info} failed in the cell number {exec_res.get('failed cell number', None)}.\n")

        # Print test outputs
        for k, v in exec_res.get('outputs', {}).items():
            message = v if isinstance(v, str) else json.dumps(v, indent=4)
            print(f"{k}:\n{message}\n")

        # Provide test conclusion
        if out_filename in pytest.out_files_info:
            print((f"Execution of {notebook_info} saved in `{out_filename}`.\n"))

        if not exec_res['failed']:
            print(f"{notebook_info} was executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed, look at `{out_filename}`.\n"

@pytest.fixture(scope="module")
def finalize_fixture():
    """ Final steps after all tests completion.

    When the last test is completed, this fixture:
        - Dump information about correspondence between saved out files and test configuration
        (the executed notebook file name and its inputs).
        - Removes `pytest.TESTS_ROOT_DIR` in case of all tests completion without failures (if needed).
    Note, if `pytest.TESTS_ROOT_DIR` is removed, then there is no need in dumping information about deleted files.
    """
    # Run all tests in the module
    yield

    # Remove pytest.TESTS_ROOT_DIR if all tests were successful
    if REMOVE_ROOT_DIR and not pytest.failed:
        shutil.rmtree(pytest.TESTS_ROOT_DIR)

    # If pytest.TESTS_ROOT_DIR exists, then dump information about out files
    else:
        dump_path = os.path.join(pytest.TESTS_ROOT_DIR, 'out_files_info.json')

        with open(dump_path, 'w') as dump_file:
            json.dump(pytest.out_files_info, dump_file, indent=4)


# Helper function
NUM_ITERATOR = iter(range(1, len(notebooks_params)+1))

def create_output_filename(input_filename, inputs):
    """ Creates output notebook filename.

    Output notebook filename consists of:
        - Executed notebook basename
        - Numeration prefix (which is equal to the test configuration execution order)
        - Processed input parameters as suffix.

    `inputs` dict is converted to a string in the following format:
    `'<key_1>_<processed_value_1>_<key_2>_<processed_value_2>_<...>'`

    Where values are processed depends on their type:
        - If value is a path string, then we add only basename without extension to the suffix.
        - In other cases we remove all non-string literals from its string representation
        and replace spaces with underscores. This is useful for cases when inputs contain lists, tuples, or dicts.

    Parameters
    ----------
    input_filename : str
        A basename of notebook for execution in tests.
    inputs : dict
        Dict of input parameters which were passed into the `input_filename` notebook.
    """
    # Prepare filename prefix which is a file num
    file_num = str(next(NUM_ITERATOR)).zfill(2)

    # Prepare filename suffix which is a short params repr
    inputs_short_repr = ""

    for param_name, param_value in inputs.items():
        param_value = str(param_value)

        if os.path.exists(param_value):
            # Cut long paths
            param_value = os.path.splitext(os.path.basename(param_value))[0]
        else:
            # Create a correct filename substring for lists, tuples, dicts
            param_value = re.sub(r'[^\w^ ]', '', param_value) # Remove all non-letter symbols except spaces
            param_value = param_value.replace(' ', '_')

        inputs_short_repr += param_name + '_' + param_value + '_'

    filename_without_ext = os.path.splitext(input_filename)[0]

    out_filename = f"{file_num}_{filename_without_ext}_out_{inputs_short_repr[:-1]}.ipynb"
    return out_filename
