from experiment import ExperimentBuilder
from string import Template


def main():
    def add(k, v): exp.add_param(k, v)
    
    exp = ExperimentBuilder(
        script=r'absolute-path-to-your-script',
        defaults=dict(
            lr=0.001,
            damp=1e-5,
            batchsize=128,
            epochs=164,
        ), CUDA_VISIBLE_DEVICES='7')

    add('use_cuda', True)
    add('lr_decay_at', [82, 123])

    exp.run(
        debug=True,
        parallelize_dict=dict(workers=5, param='seed', values=[0,1,2,3,4,5]),
        param_name_for_exp_root_folder='root_folder',
        exp_folder=f'abs-exp-path',
        exp_name=Template('lr=${lr}_damp=${damp}_batchsize=${batchsize}_epochs=${epochs}_seed=${seed}')
    )


if __name__ == '__main__':
    main()
