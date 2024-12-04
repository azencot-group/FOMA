dataset_defaults = {
    'poverty': {
        'optimiser': 'Adam',
        'pretrain_iters': 0,
        'meta_lr': 0.1,
        'meta_steps': 5,
        'selection_metric': 'r_wg',
        'reload_inner_optim': True,
        'print_iters': 350,
        'scheduler': 'StepLR',
        'scheduler_kwargs': {'gamma': 0.96},
        'command_args': {
            'estimate_id': 0,
            'rho': 0.875,
            'batch_selection': 'knnp',
            'lr': 5e-3,
            'foma_input': 0,
            'foma_latent': 1,
            'alpha': 1,
            'batch_size': 32,
            'num_epochs': 50,
            'small_singular': 1,
        },
    }
}
