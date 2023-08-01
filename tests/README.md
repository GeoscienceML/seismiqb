# A short instruction for adding new test notebooks

## Notebook preparation

Before adding a new test notebook to the list of automatically executed tests, make the following **preparations**:

1. If the test notebook saves any files, it is highly recommended to use relative paths and create a test's **own directory** for saving files.

The `TESTS_ROOT_DIR` (`'seismiqb\tests\test_root_dir_*'`) is a directory from which all relative paths start. `TESTS_ROOT_DIR` is shared between all test notebooks and creating separate test directories inside it prevents from mixing up files from different tests.

2. All **externally parameterized variables** must be initialized with default values in the first or second notebook cells. All of the actual testing must be done after the cell number 2.

This is because the `run_notebook_test.py` inserts a new cell with parameters initialization between cells number 2 and 3.

So, the recommended notebook structure is:
* **Cell #1**: necessary imports.
* **Cell #2**: parameters initialization.
* **Cells #3+**: tests and additional code.

## Adding a new notebook to the list of automatically executed tests

Once the notebook is prepared, it can be added to the list of automatically executed notebooks.

In order to do that provide a `(notebook_path, params_dict)` tuple into the `notebooks_params` variable inside the `run_notebook_test.py`.

The `params_dict` is a dictionary with optional `'inputs'` and `'outputs'` keys:
* If the test notebook must be executed with **new parameters values**, just add them in the `'inputs'` in the dictionary format `{'parameter_name': 'parameter_value'}`.
* If it is important to **print into the terminal** some variables values from the executed notebook (such as log messages or timings), add them in the `'outputs'` in the list format `['notebook_variable_name_1', 'notebook_variable_name_2']`.

```python
notebooks_params = (
    ('path/to/the/test_notebook.ipynb', {}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}}),
    ('path/to/the/test_notebook.ipynb', {'outputs': ['message', 'timings']}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}, 'outputs': ['message']})
)
```

That's all, now you know how to add new tests!

# Additional information

## Good practices for test notebooks

Some recommended optional practices for creating good test notebooks are recorded in the `seismiqb/tests/template_test.ipynb`.

## More about the `TESTS_ROOT_DIR`

`TESTS_ROOT_DIR` is a directory for saving files for **all running tests**. A new `TESTS_ROOT_DIR` is created for each tests run.
If tests executed locally (**not** on the GitHub), then it is a directory in the format: `'seismiqb/tests/tests_root_dir_*'`.

* If all tests are executed without any failures and `REMOVE_ROOT_DIR` is True, the `TESTS_ROOT_DIR` is removed after all tests execution.
* If there are any failures in tests and/or `REMOVE_ROOT_DIR` is False, the `TESTS_ROOT_DIR` is not removed after all tests execution.
In this case saved notebooks can be checked to find out the failure reason.

## More about the `notebooks_params` variable

The important details are:
* Notebooks are executed in the order they are defined in the `notebooks_params`.
* For the notebook execution with different parameters configurations, all of them must be provided into the `notebooks_params`:

```python
notebooks_params = (
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'sgy'}}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'blosc'}}),
    ('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': 'hdf5'}}),
)
```

```python
notebooks_params = (
    *[('path/to/the/test_notebook.ipynb', {'inputs': {'FORMAT': data_format}}) for data_format in ['sgy', 'blosc', 'hdf5']],
)
```

## More about terminal output message

The `run_notebook_test.py` provides in the terminal output next information:
* Error traceback and additional error info (if there is any failure in the test notebook). The additional info is: the notebook file name and the failed cell number.
* Notebook's `'outputs'` (if any is provided for the notebook into the `notebooks_params`).
* Test conclusion: whether the notebook with tests failed or not.

One noticeable moment, the message `Notebook execution failed` is printed in two cases:

1. There is any **failure** in the notebook. Then there must be an error traceback above this message:

```python
run_notebook_test.py ---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_38767/262262589.py in <module>
----> 1 assert False, "Test failure for the example"

AssertionError: Test failure for the example
Notebook execution failed


`example_test.ipynb` failed in the cell number 6.
```

2. The notebook **wasn't executed**. In this case there are no traceback above this message. The reason for this situation is some internal execution error such as out of memory.

```python
run_notebook_test.py ---------------------------------------------------------------------------
Notebook execution failed
```

## Correspondence between output file names and test configurations

Output file names processed from the execution count, the executed notebook name and passed inputs into it.
Correspondence between out file name and its test configuration is saved in `seismiqb/tests/tests_root_dir_*/out_files_info.json`.

Note, this file contains information only about **saved** executed notebooks. If the `REMOVE_EXTRA_FILES` flag is True and notebook is executed without any failure, then it is not saved.

Example of `out_files_info.json`:

```json
{
    "02_geometry_test_01_preparation_out_FORMATS_sgy_hdf5.ipynb": {
        "filename": "geometry_test_01_preparation.ipynb",
        "inputs": {
            "FORMATS": [
                "sgy",
                "hdf5"
            ]
        }
    },
    "03_geometry_test_02_data_format_out_FORMAT_sgy.ipynb": {
        "filename": "geometry_test_02_data_format.ipynb",
        "inputs": {
            "FORMAT": "sgy"
        }
    },
    "04_geometry_test_02_data_format_out_FORMAT_hdf5.ipynb": {
        "filename": "geometry_test_02_data_format.ipynb",
        "inputs": {
            "FORMAT": "hdf5"
        }
}
```

In this example executions number 2,3 and 4 was executed with failures. Other executions did **not** fail and therefore, they weren't saved.
