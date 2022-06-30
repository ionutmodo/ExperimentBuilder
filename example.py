from experiment import ExperimentBuilder
from string import Template


def main():
    def add(k, v):
        exp.add_param(k, v)
    
    exp = ExperimentBuilder(
        script=r'absolute-path-to-your-script',
        defaults=dict(
            lr=0.001,
            damp=1e-5,
            use_cuda=True,
            lr_decay_at=[82, 123],
            batchsize=128,
            epochs=164,
        ), CUDA_VISIBLE_DEVICES='0')

    for lr in [1e-4, 1e-3, 1e-2]:
        print(f'lr={lr}')

        add('lr', lr)
        exp.run(
            debug=True,
            parallelize_dict=dict(workers=5, param='seed', values=[0, 1, 2, 3, 4]),
            param_name_for_exp_root_folder='root_folder',
            exp_folder=f'abs-exp-path',
            exp_name=Template('lr=${lr}_damp=${damp}_batchsize=${batchsize}_epochs=${epochs}_seed=${seed}')
        )


if __name__ == '__main__':
    main()

"""
lr=0.0001
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 0 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=0
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 1 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=1
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 2 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=2
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 3 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=3
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.0001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 4 --root_folder abs-exp-path\lr=0.0001_damp=1e-05_batchsize=128_epochs=164_seed=4

lr=0.001
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 0 --root_folder abs-exp-path\lr=0.001_damp=1e-05_batchsize=128_epochs=164_seed=0
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 1 --root_folder abs-exp-path\lr=0.001_damp=1e-05_batchsize=128_epochs=164_seed=1
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 2 --root_folder abs-exp-path\lr=0.001_damp=1e-05_batchsize=128_epochs=164_seed=2
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 3 --root_folder abs-exp-path\lr=0.001_damp=1e-05_batchsize=128_epochs=164_seed=3
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.001 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 4 --root_folder abs-exp-path\lr=0.001_damp=1e-05_batchsize=128_epochs=164_seed=4

lr=0.01
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 0 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=0
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 1 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=1
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 2 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=2
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 3 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=3
CUDA_VISIBLE_DEVICES=0 python absolute-path-to-your-script --lr 0.01 --damp 1e-05 --use_cuda --lr_decay_at 82 123 --batchsize 128 --epochs 164 --seed 4 --root_folder abs-exp-path\lr=0.01_damp=1e-05_batchsize=128_epochs=164_seed=4
"""