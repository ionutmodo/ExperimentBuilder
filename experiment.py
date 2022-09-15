import multiprocessing as mp
from string import Template
from itertools import product
from copy import deepcopy
from tools import *


def waiting_worker(params):
    # cmd, root, cmd_dict, gpu_processes_count, scheduling['gpus'], scheduling['max_jobs_per_gpu'], scheduling['distributed_training']
    cmd, root, cmd_dict, gpu_processes_count, gpus, max_jobs, dist_train = params
    n_gpus = len(gpus)
    time.sleep(random.randint(1, n_gpus * max_jobs))

    if not dist_train:
        while True:
            sorted_items = sorted(gpu_processes_count.items(), key=lambda item: item[1])
            gpu, count = sorted_items[0] # sort ASC by processes count

            # if there are multiple GPUs with minimal number of processes, then pick a random GPU from them
            i = 1
            while i < n_gpus and sorted_items[i][0] == count:
                i += 1
            if count < max_jobs:
                gpu = random.choice([c for g, c in sorted_items[:i]])
                gpu_processes_count[gpu] += 1
                break

            print(f'All GPUs in have {max_jobs} jobs, waiting 60 seconds...')
            time.sleep(60)

    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, 'arguments.txt'), 'w') as w:
        for k, v in cmd_dict.items():
            if k.startswith('_'):
                w.write(f'{k[1:]}={v}\n')

    if dist_train:
        cvd = f'CUDA_VISIBLE_DEVICES={",".join(map(str, gpus))}'
    else:
        cvd = f'CUDA_VISIBLE_DEVICES={gpu}'

    cmd = f'{cvd} {cmd}'

    print(cmd)
    os.system(cmd)

    if not dist_train:
        gpu_processes_count[gpu] -= 1


class ExperimentBuilder:
    def __init__(self, script, defaults=None, verbose=True):
        """
        :param script: path to script, rooted in home directory (it's automatically inserted as prefix)
        :param defaults: default cmd arguments that usually stay fixed
        :param CUDA_VISIBLE_DEVICES: value to initialize CUDA_VISIBLE_DEVICES env variable
        """
        self.script = script
        self.verbose = verbose
        self.exp_name_template = None
        self.exp_folder_template = None

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
                setattr(self, f'_{name}', value)
            elif isinstance(value, Template):
                setattr(self, f'template_{name}', deepcopy(value))
                setattr(self, f'_{name}', None)
                # setattr(self, f'_{name}', self._fill_template(value)) # to avoid dict-changed-size error
            else:
                setattr(self, f'_{name}', value)


    def run(self,
            exp_folder: Template,
            exp_name: Template,
            param_name_for_exp_root_folder: str,
            scheduling: dict,
            debug: bool = False):
        """
        :param exp_folder: absolute path of the root folder where you want your experiments to be
        :param exp_name: template used to generate experiment name
            The placeholders name should be the parameter names added using add_param method
        :param param_name_for_exp_root_folder: the cmd argument name for the output directory
        :param scheduling: a dictionary containing keys `gpus`, `max_jobs_per_gpu`, `params_values`
            - `gpus` is a list containing IDs of GPUs you want to run the tasks on
            - `distributed_training` a boolean indicating whether the experiment uses DataParallel or not
            - `max_jobs_per_gpu` specifies how many processes should run on each GPU at most (num_workers = len(gpus) * max_jobs_per_gpu)
            - `param_values` is the dictionary that contains values for multiple parameters (the cartesian product will be computed)
        :param debug: print commands if True, run commands if False
        """
        assert 'gpus' in scheduling.keys(), 'scheduling requires `gpu` key'
        assert 'params_values' in scheduling.keys(), 'scheduling requires `params_values` key'
        assert 'max_jobs_per_gpu' in scheduling.keys(), 'scheduling requires `max_jobs_per_gpu` key'
        assert 'distributed_training' in scheduling.keys(), 'scheduling requires `distributed_training` key'
        """
            scheduling['gpus']
            scheduling['max_jobs_per_gpu']
            scheduling['params_values']
            scheduling['distributed_training']
        """
        n_gpus = len(scheduling['gpus'])
        if scheduling['distributed_training']: # use all GPUs for a single run (distributed training)
            n_workers = scheduling['max_jobs_per_gpu']
        else: # use GPUs to run one experiment per GPU
            n_workers = n_gpus * scheduling['max_jobs_per_gpu']

        self.exp_name_template = deepcopy(exp_name)
        self.exp_folder_template = deepcopy(exp_folder)

        if not on_windows():
            os.system('clear')

        cmds = []
        cmds_dict = []
        root_folders = []

        params = list(scheduling['params_values'].keys())

        cart_prod = list(product(*list(scheduling['params_values'].values())))
        for i, values in enumerate(cart_prod):
            for k, v in zip(params, values):
                self.add_param(k, v)
            # after filling in the values for HPO, go through all templated fields and fill them with the new values
            for k, v in self.__dict__.items():
                if k.startswith('template_'):
                    tmpl_filled = self._fill_template(v)
                    self.__dict__[k.replace('template', '')] = tmpl_filled # only replace "template", keep "_"

            root_folder = self._create_root_arg(
                param_name_for_exp_root_folder,
                self.exp_folder_template,
                self.exp_name_template)

            p = {k: v for k, v in self.__dict__.items() if k.startswith('_')}
            cmds_dict.append(p)
            root_folders.append(root_folder)
            cmds.append(self._build_command())
        if debug:
            for cmd in cmds:
                print(cmd)
        else:
            manager = mp.Manager()
            gpu_processes_count = manager.dict()
            for gpu in scheduling['gpus']:
                gpu_processes_count[gpu] = 0

            with mp.Pool(processes=n_workers) as pool:
                pool.map(
                    func=waiting_worker,
                    iterable=[
                        (cmd, root, cmd_dict, gpu_processes_count, scheduling['gpus'], scheduling['max_jobs_per_gpu'], scheduling['distributed_training'])
                        for cmd, root, cmd_dict, in zip(cmds, root_folders, cmds_dict)
                    ])

    def _create_root_arg(self, param_name_for_exp_root_folder, exp_folder, exp_name):
        exp_root_folder = os.path.join(self._fill_template(exp_folder), self._fill_template(exp_name))
        self.add_param(param_name_for_exp_root_folder, exp_root_folder)
        return exp_root_folder

    def _fill_template(self, template):
        """
        This method fills in the `template` given as parameter with values stored in `self.__dict__`.
        If the template uses a variable which is not in `self.__dict__` yet, the method returns the template again
        because that parameter is expected to be set in `parallelize_dict`. This way, this parameter is not lost
        :param template: the template to be filled. if it's string, then it's immediately returned
        :return: a string containing the template with substitutions or the same template
        """
        if isinstance(template, str):
            return template
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
        return f'python {self.script} {params}'

    def __getattr__(self, item):
        return self.__dict__[f'_{item}']
