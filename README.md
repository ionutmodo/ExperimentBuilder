# ExperimentBuilder

---

This project is used as an alternative to bash scripts to run Python programs with command line arguments managed by `argparse`.

How to use ExperimentBuilder?
-----------------------------

Suppose you have both your project and the `ExperimentBuilder` project in the same folder on the disk:

- `~/workplace/Application/`
  - `main.py`
  - `run_exp.py`
- `~/workplace/ExperimentBuilder/experiment.py`

Your `main.py` contains the program that performs your task, and `run_exp.py` should build the command line arguments using `ExperimentBuilder`. In your `run_exp.py` script, you should link the `ExperimentBuilder` project, such that you can use the module `experiment.py`:

```python
import os, sys

def link_experiment_library():
    home = os.path.expanduser('~')
    exp_lib_path = os.path.join(home, 'workplace', 'ExperimentBuilder')
    sys.path.append(exp_lib_path)
```

After that, you can build your script to run experiment, similar to the file [experiment.py](https://github.com/ionutmodo/ExperimentBuilder/blob/main/example.py) in this repository.

Details about `ExperimentBuilder` class in `experiment.py` file:
----------------------------------------------------------------

The constructor requires the following parameters:
- `script`: the absolute path to the program that will be run, e.g. `~/workplace/Application/main.py`
- `defaults`: a dictionary containing default parameters for your program, which will not change if you perform, for example, hyper-parameter tuning, such as `epochs`
<!-- - `CUDA_VISIBLE_DEVICES`: similar to the environment variable in Linux -->
- `verbose`: boolean indicating whether you want to print information about the folders that are created when calling `run` method

Once the `ExperimentBuilder` object is created, you can add command line arguments on the fly using `add_param` method as key and value pairs. They will be converted automatically to strings and will be added to the project using the `getattr` builtin function, preceded by the underscore prefix (to be able to differentiate them from other attributes in the object). There is also support for templates in the `add_param` method. If you add such a parameter, you can use the previously added parameter values to create the value of the templated parameter.

There is also support for boolean command line arguments. For example, if you use `add_param('normalize', True)`, then it will only add the `--normalize` argument. If you use `add_param('normalize', False)`, then no argument will be added to the command line.

After you added all required arguments (either in constructor using `defaults` or by calling `add_param` method), you should call the `run` method, with the following parameters:
- `scheduling`: a dictionary containing the following keys:
  - `gpus` is a list containing IDs of GPUs you want to run the tasks on
  - `distributed_training` a boolean indicating whether the experiment uses DataParallel or not
  - `max_jobs_per_gpu` specifies how many processes should run on each GPU at most (num_workers = len(gpus) * max_jobs_per_gpu)
  - `param_values` is the dictionary that contains values for multiple parameters (the cartesian product will be computed)
  - see [experiment.py](https://github.com/ionutmodo/ExperimentBuilder/blob/main/example.py) for more details
- `exp_folder`: the root folder where you will store all experiments
- `exp_name`: the name of your experiment, which will be a folder saved at the path given by the `exp_folder` parameter. This is a template that allows you to use hyperparameter values in the name. For example, if you called `add_param('lr', 0.1)`, then you can use ```exp_name=Template('lr=${lr}')``` and the folder will be named `lr=0.1`
- `param_name_for_exp_root_folder`: it is expected that your `~/workplace/Application/main.py` script require a parameter that specifies a root folder on the disk where you will save all your experiments to (suppose it's called `root_folder`). In this case, you must set `param_name_for_exp_root_folder='root_folder'` such that, in the `run` method, it will be given the value `os.path.join(exp_folder, exp_name)` when you run the script
- `debug`: set it to True if you only want to print the commands that the `ExperimentBuilder` builds. Set it to False in order to actually run those commands in a Linux environment

The command line arguments will be saved in the experiment folder in the file `arguments.txt`.

An example is provided in the file [experiment.py](https://github.com/ionutmodo/ExperimentBuilder/blob/main/example.py) and you can actually see the output for that program at the bottom of the script.