import multiprocessing as mp
from string import Template
import sys
import os


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

        for k, v in defaults.items():
            self.add_param(k, v)

    def add_param(self, name, value):
        if isinstance(value, list):
            value = ' '.join(map(str, value))
        setattr(self, f'_{name}', value)

    def run(self, exp_folder: str, exp_name: Template, param_name_for_exp_root_folder: str, parallelize_dict:dict=None, debug:bool=False):
        """
        :param exp_folder: absolute path of the root folder where you want your experiments to be
        :param exp_name: template used to generate experiment name
            The placeholders name should be the parameter names added using add_param method
        :param param_name_for_exp_root_folder: the cmd argument name for the output directory
        :param parallelize_dict: a dictionary. Example {'workers': int, 'param': str, 'values': list)
            - workers is the number of workers for the process pool. Set it to zero to launch one process per parameter
            - param is the parameter to parallelize over
            - values is the list of values for the parameter to be run in parallel using multiprocessing
            This is teste only for CPU device
        :param debug:
        :return:
        """
        if ('linux' in sys.platform) or ('darwin' in sys.platform):
            os.system('clear')

        if parallelize_dict is None:  # run a single process
            self._create_folder_arg_then_makedir_then_write_parameters(param_name_for_exp_root_folder, exp_folder, exp_name)
            cmd = self._build_command()
            if debug:
                print(cmd)
            else:
                os.system(cmd)

            print('EXPERIMENT ENDED')
            print(cmd)
        else:
            cmds = []
            for v in parallelize_dict['values']:
                self.add_param(parallelize_dict['param'], v)
                self._create_folder_arg_then_makedir_then_write_parameters(param_name_for_exp_root_folder, exp_folder, exp_name)
                cmds.append(self._build_command())
            if debug:
                for cmd in cmds:
                    print(cmd)

            n_procs = parallelize_dict['workers'] if parallelize_dict['workers'] > 0 else len(parallelize_dict['values'])
            with mp.Pool(processes=n_procs) as pool:
                pool.map(func=os.system, iterable=cmds)

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
        return template.substitute(**{
            key[1:]: val
            for key, val in self.__dict__.items()
            if key.startswith('_')
        })

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
