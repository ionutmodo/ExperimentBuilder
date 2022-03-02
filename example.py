from experiment import ExperimentBuilder


def main():
    eb = ExperimentBuilder(
        script=r'workplace/M-FAC_extensions/main.py',
        dataset='/home/Datasets/cifar10',
        defaults=dict(
            weightdecay=1e-4,
            lr_decay=0.1,
            lr=0.001,
            damp=1e-5,
            seed=0,
            batchsize=128,
            epochs=164,
            lr_decay_at=[82, 123]),
        CUDA_VISIBLE_DEVICES='7')

    eb.add_param('grads', 128)
    eb.add_param('momentum', 0)
    eb.add_param('optim', 'kmfac')
    eb.add_param('pca_optim_method', 'none')
    eb.add_param('pca_optim_warmup_steps', 0)

    for model in ['resnet20']:
        for k in [2048]:
            for cg in [0]:
                eb.add_param('model', model)
                eb.add_param('k', k)
                eb.add_param('compress_gradients', cg)
                eb.run(
                    exp_folder=f'workplace/experiments_resnet_cifar10/MDA/extra_exps_28feb/'
                               f'{eb.optim}-{eb.pca_optim_method}-{eb.pca_optim_warmup_steps}',
                    exp_name='-'.join([
                        f'optim={eb.optim}+{eb.pca_optim_method}+{eb.pca_optim_warmup_steps}+{eb.compress_gradients}',
                        f'model={eb.model}',
                        f'k={eb.k}',
                        f'mom={eb.momentum}',
                        f'wd={eb.weightdecay}',
                        f'lrd={eb.lr_decay}',
                        f'lr={eb.lr}',
                        f'damp={eb.damp}',
                        f'grads={eb.grads}',
                        f'seed={eb.seed}']))


if __name__ == '__main__':
    main()
