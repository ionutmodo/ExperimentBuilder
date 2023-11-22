from string import Template
import os
import sys


def link_experiment_library():
    home = os.path.expanduser('~')
    exp_lib_path = os.path.join(home, 'workplace', 'ExperimentBuilder')
    sys.path.append(exp_lib_path)


def main():
    link_experiment_library()
    from experiment import ExperimentBuilder

    def add(k, v):
        exp.add_param(k, v)
    
    exp = ExperimentBuilder(
        script=r'./tmp/empty.py',
        defaults=dict(
            lr=0.001,
            batchsize=128,
            epochs=100,
            lr_decay_at=[82, 123],
        ),
        verbose=True)

    exp.add_param('TEST', Template('lr=${lr}_epochs=${epochs}_batchsize=${batchsize}_seed=${seed}'))

    exp.run(
        debug=False,
        torchrun=True,
        scheduling=dict(
            distributed_training=True,
            gpus=[4, 5, 6, 7],
            max_jobs_per_gpu=4,
            params_values=dict(seed=[111, 222], optim=['adam', 'sgd'])),
        param_name_for_exp_root_folder='root_folder',
        exp_folder=Template(os.path.join('./tmp', 'lr=${lr}_batchsize=${batchsize}_epochs=${epochs}_seed=${seed}_${optim}')))


if __name__ == '__main__':
    main()

"""
Below you can find the output of this program:

lr=0.0001
Added parameter root_folder=abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=0
Created folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=0 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=1
Created folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=1 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=2
Created folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=2 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=3
Created folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=3 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=4
Created folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=4 and wrote command arguments to "arguments.txt" file inside it

CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 0 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=0
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 1 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=1
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 2 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=2
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 3 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=3
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 4 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=4
--------------------------------------------------
lr=0.01
Added parameter root_folder=abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=0
Created folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=0 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=1
Created folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=1 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=2
Created folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=2 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=3
Created folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=3 and wrote command arguments to "arguments.txt" file inside it

Added parameter root_folder=abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=4
Created folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=4 and wrote command arguments to "arguments.txt" file inside it

CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 0 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=0
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 1 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=1
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 2 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=2
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 3 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=3
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 4 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=4
--------------------------------------------------
"""