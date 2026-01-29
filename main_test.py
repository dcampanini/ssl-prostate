import os
import warnings

import torch
import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb

import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers
from datetime import datetime


def main():

    args = get_conf()

    args.test = False

    # set seed if required
    set_seed(args.seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, 
                nprocs=ngpus_per_node, 
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    dt_string = dt_string.replace(":","_")

    args.output_dir = f"{args.output_dir}_{dt_string}"
    args.ckpt_dir = args.output_dir

    # init trainer
    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    for f in [args.fold_id]:
        trainer = trainer_class(args)
        
        if args.rank == 0 and not args.disable_wandb:
            # generate an ID for the wandb run
            if args.wandb_id is None:
                args.wandb_id = wandb.util.generate_id()

            run = wandb.init(project=args.wandb_proj or f"{args.proj_name}_{args.dataset}", 
                            name=args.run_name, 
                            config=vars(args),
                            id=args.wandb_id,
                            resume='allow',
                            dir=args.output_dir)

        # create model
        trainer.build_model()
        # create optimizer
        trainer.build_optimizer()
        trainer.build_dataloader()
        
        # train or test
        if args.mass_testing:
            trainer.evaluate_test_all()

        if args.rank == 0 and not args.disable_wandb:
            run.finish()

        if f == 1:
            break


if __name__ == '__main__':
    main()
