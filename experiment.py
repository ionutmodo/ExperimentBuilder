import multiprocessing as mp
from string import Template
from itertools import product
import sys
import os

from tools import read_yaml, wait_for_processes, wait_for_gpus_of_user


def waiting_worker(params):
    cmd, wait_for_gpus, gpus2wait4, pids2wait4 = params
    if wait_for_gpus:  # waiting for GPUs has higher priority
        wait_for_gpus_of_user(gpus2wait4)
    else:
        wait_for_processes(pids2wait4)
    os.system(cmd)


class ExperimentBuilder:
    def __init__(self, script, defaults, CUDA_VISIBLE_DEVICES, verbose=True):
        """
        @param script: path to script, rooted in home directory (it's automatically inserted as prefix)
        @param defaults: default cmd arguments that usually stay fixed
        @param CUDA_VISIBLE_DEVICES: value to initialize CUDA_VISIBLE_DEVICES env variable
        """
        self.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        self.script = script
        self.verbose = verbose

        if defaults is not None:
            for k, v in defaults.items():
                self.add_param(k, v)

    def add_from_yaml(self, yaml_file):
        data = read_yaml(yaml_file)
        for k, v in data.items():
            self.add_param(name=k, value=v)

    def add_param(self, name, value):
        """
        Adds a parameter to the command line args in the form "--name value" (roughly)
        :param name: The name of the parameter, which will be preceded by two dashes ("--")
        :param value: The value for the parameter. If it's a Template, it will be filled in with the values of already existing parameters
        :return:
        """
        if value is not None:
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            elif isinstance(value, Template):
                value = self._fill_template(value)

            setattr(self, f'_{name}', value)

    def run(self,
            exp_folder: str,
            exp_name: Template,
            param_name_for_exp_root_folder: str,
            parallelize_dict: dict = None,
            debug: bool = False,
            wait_for_pids: dict = None,
            wait_for_gpus: bool = True):
        """
        :param exp_folder: absolute path of the root folder where you want your experiments to be
        :param exp_name: template used to generate experiment name
            The placeholders name should be the parameter names added using add_param method
        :param param_name_for_exp_root_folder: the cmd argument name for the output directory
        :param parallelize_dict: a dictionary. Example {'workers': int, 'param_values': {'p1': L1, 'p2': L2}})
            - workers is the number of workers for the process pool. Set it to zero to launch one process per parameter
            - param_values is the dictionary that may contain values for multiple parameters (the cartesian product will be computed)
        :param debug: print commands if True, run commands if False
        :param wait_for_pids: a dictionary containing two keys: prefix and suffixes. For example, if you want to wait for processes 1230, 1231, 123, 1233,
        you should set wait_for_pids={prefix=123, suffixes=[0,1,2,3]} (all should be ints for simplicity). The result is a list containing the PIDs of
        processes that need to finish before running this script; if None, then start the current process(es) right away.
        This is useful when another process is currently using the GPUs you also want to use
        :return:
        """
        gpus2wait4 = list(map(int, self.CUDA_VISIBLE_DEVICES.split(',')))
        pids2wait4 = None
        if wait_for_pids is not None:
            pids2wait4 = []
            p = wait_for_pids['prefix']
            for s in wait_for_pids['suffixes']:
                pids2wait4.append(int(f'{p}{s}'))

        if ('linux' in sys.platform) or ('darwin' in sys.platform):
            os.system('clear')

        if parallelize_dict is None:  # run a single process
            self._create_folder_arg_then_makedir_then_write_parameters(param_name_for_exp_root_folder, exp_folder, exp_name)
            cmd = self._build_command()
            if debug:
                print(cmd)
            else:
                if wait_for_gpus:  # waiting for GPUs has higher priority
                    wait_for_gpus_of_user(gpus2wait4)
                else:
                    wait_for_processes(pids2wait4)
                os.system(cmd)

            print('EXPERIMENT ENDED')
            print(cmd)
        else:
            cmds = []
            n_workers = parallelize_dict['workers']
            params_values_dict = parallelize_dict['params_values']
            params = list(params_values_dict.keys())
            n_params = len(params) # how many parameters we have: seed, optim, etc

            # cartesian product of all lists of values for each parameter
            # from documentation: product(L1, L2)

            # v = params_values_dict.values()
            # v_list = list(v)
            # cart_prod = product(*v_list)
            # cart_prod_list = list(cart_prod)
            # for values in cart_prod_list:

            for values in product(*list(params_values_dict.values())):
                for i in range(n_params):
                    self.add_param(params[i], values[i])
                self._create_folder_arg_then_makedir_then_write_parameters(param_name_for_exp_root_folder, exp_folder, exp_name)
                cmds.append(self._build_command())
            # for v in parallelize_dict['values']:
            #     self.add_param(parallelize_dict['param'], v)
            #     self._create_folder_arg_then_makedir_then_write_parameters(param_name_for_exp_root_folder, exp_folder, exp_name)
            #     cmds.append(self._build_command())
            if debug:
                for cmd in cmds:
                    print(cmd)
            else:

                with mp.Pool(processes=n_workers) as pool:
                    pool.map(func=waiting_worker, iterable=[(cmd, wait_for_gpus, gpus2wait4, pids2wait4) for cmd in cmds])

    def _create_folder_arg_then_makedir_then_write_parameters(self, param_name_for_exp_root_folder, exp_folder, exp_name):
        exp_root_folder = os.path.join(exp_folder, self._fill_template(exp_name))
        os.makedirs(exp_root_folder, exist_ok=True)
        self.add_param(param_name_for_exp_root_folder, exp_root_folder)

        with open(os.path.join(exp_root_folder, 'arguments.txt'), 'w') as w:
            for k, v in self.__dict__.items():
                if k.startswith('_'):
                    w.write(f'{k[1:]}={v}\n')
        if self.verbose:
            print(f'Added parameter {param_name_for_exp_root_folder}={exp_root_folder}')
            print(f'Created folder {exp_root_folder} and wrote command arguments to "arguments.txt" file inside it')
            print()

    def _fill_template(self, template):
        """
        This method fills in the `template` given as parameter with values stored in `self.__dict__`.
        If the template uses a variable which is not in `self.__dict__` yet, the method returns the template again
        because that parameter is expected to be set in `parallelize_dict`. This way, this parameter is not lost
        :param template: the template to be filled
        :return: a string containing the template with substitutions or the same template
        """
        try:
            d = {}
            for key, val in self.__dict__.items():
                if key.startswith('_'):
                    d[key[1:]] = val

            substituted = template.substitute(**d)
            return substituted
        except KeyError as e:
            print(f'[TemplateError] {str(e)}, {e.__cause__}')
            return template

    def _build_command(self):
        cvd = 'CUDA_VISIBLE_DEVICES'
        params = []
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                if isinstance(v, bool):  # we have a parameter that does not have a value, but its presence or absence means True or False
                    if v:
                        params.append(f'--{k}')
                elif isinstance(v, Template):
                    # elif isinstance(v, str) and '${' in v:
                    params.append(f'--{k} {self._fill_template(v)}')
                else:
                    params.append(f'--{k} {str(v)}')
        params = ' '.join(params).replace('--_', '--')
        return f'{cvd}={os.environ[cvd]} python {self.script} {params}'

    def __getattr__(self, item):
        return self.__dict__[f'_{item}']
