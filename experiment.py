import multiprocessing as mp
import sys
import os


class ExperimentBuilder:
    def __init__(self, script, defaults, CUDA_VISIBLE_DEVICES):
        """
        @param script: path to script, rooted in home directory (it's automatically inserted as prefix)
        @param defaults: default cmd arguments that usually stay fixed
        @param CUDA_VISIBLE_DEVICES: value to initialize CUDA_VISIBLE_DEVICES env variable
        """
        self.devices = CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        self.script = script

        for k, v in defaults.items():
            self.add_param(k, v)

    def add_param(self, name, value):
        if isinstance(value, list):
            value = ' '.join(map(str, value))
        setattr(self, f'_{name}', value)

    def run(self, exp_folder, exp_name, param_name_for_exp_root_folder, parallelize_dict=None, debug=False):
        """
        :param exp_folder: absolute path of the root folder where you want your experiments to be
        :param exp_name: folder name for the experiment
        :param param_name_for_exp_root_folder: the cmd argument name for the output directory
        :param parallelize_dict: a dictionary. Example {'workers': int, 'param': str, 'values': list)
            - workers is the number of workers for the process pool
            - param is the parameter to parallelize over
            - values is the list of values for the parameter to be run in parallel using multiprocessing
            This is teste only for CPU device
        :param debug:
        :return:
        """
        exp_root_folder = os.path.join(exp_folder, exp_name)
        # log_file_path = os.path.join(exp_root_folder, f'output_{exp_name}.txt')

        os.makedirs(exp_root_folder, exist_ok=True)

        self.add_param(param_name_for_exp_root_folder, exp_root_folder)
        # self.add_param('log_file_path', log_file_path)

        if ('linux' in sys.platform) or ('darwin' in sys.platform):
            os.system('clear')

        if parallelize_dict is None:  # run a single process
            cmd = self._build_command()
            if debug:
                print(cmd)
            else:
                os.system(cmd)

            print('ended', exp_name)
            print(cmd)
        else:
            cmds = []
            for v in parallelize_dict['values']:
                self.add_param(parallelize_dict['param'], v)
                cmds.append(self._build_command())
            if debug:
                for cmd in cmds:
                    print(cmd)
            with mp.Pool(processes=parallelize_dict['workers']) as pool:
                pool.map(func=os.system, iterable=cmds)


    def _build_command(self):
        cvd = 'CUDA_VISIBLE_DEVICES'
        params = []
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                if isinstance(v, bool): # we have a parameter that does not have a value, but its presence or absence means True or False
                    if v:
                        params.append(f'--{k}')
                else:
                    params.append(f'--{k} {str(v)}')
        params = ' '.join(params).replace('--_', '--')
        return f'{cvd}={os.environ[cvd]} python {self.script} {params}'

    def __getattr__(self, item):
        return self.__dict__[f'_{item}']
