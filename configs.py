import os
import time
from argparse import ArgumentParser


def parse_seg_args():
    parser = ArgumentParser()
    parser.add_argument('--comment', type=str, default='', help='save comment')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers to load data')
    parser.add_argument('--amp', action='store_true', help='using mixed precision')

    # split
    parser.add_argument('--split_file', type=str, required=True, help='path to json file that contains data split info')
    
    # ddp (multi-gpu support)
    parser.add_argument('--distributed', action='store_true', help='use DDP training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='DDP backend')
    parser.add_argument('--local_rank', type=int, default=0, help='DDP local rank (auto-set by torchrun)')

    # path & dir
    parser.add_argument('--exp_dir', type=str, default='exps', help='experiment dir')
    parser.add_argument('--save_freq', type=int, default=10, help='model save frequency (epoch)')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency (iteration)')

    # data
    parser.add_argument('--dataset', type=str, default='brats2024', help='dataset hint', 
        choices=['brats2024'])
    parser.add_argument('--data_root', type=str, default='data/', help='root dir of dataset')
    parser.add_argument('--input_channels', '--n_views', type=int, default=4, 
        help="#channels of input data, equal to #encoders in multiencoder unet and" \
             "#view in multiview contrastive learning")

    # data augmentation
    parser.add_argument('--patch_size', type=int, default=128, help='patch size')
    parser.add_argument('--pos_ratio', type=float, default=1.0, 
        help="prob of picking positive patch (center in foreground)")
    parser.add_argument('--neg_ratio', type=float, default=1.0, 
        help="prob of picking negative patch (center in background)")

    # optimize
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optim', type=str, default='adamw', help='optimizer', 
        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--beta1', default=0.9, type=float, metavar='M', 
        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, metavar='M', help='beta2 for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--scheduler', type=str, default='none', help='scheduler',
        choices=['warmup_cosine', 'cosine', 'step', 'poly', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warm up epochs')
    parser.add_argument('--milestones', type=int, nargs="+", default=[60, 80], 
        help='milestones for multistep decay')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
        help='decay factor for multistep decay')
    parser.add_argument('--clip_grad', action='store_true', help='whether to clip gradient')

    parser.add_argument('--model', type=str, default='unet',
        choices=['unet', 'swin_unetr'],
        help='Top-level model architecture. Used for exp_dir naming and (if you wish) '
             'to dispatch model building. The trainer scripts (train_ddp_unet.py / '
             'train_ddp_swin.py) hardcode their model, so this flag mainly tags exp_dir.')

    # u-net
    parser.add_argument('--unet_arch', type=str, default='unet', 
        choices=['unet', 'multiencoder_unet'], help='Architecuture of the U-Net')
    parser.add_argument('--block', type=str, default='plain', choices=['plain', 'res'],
        help='Type of convolution block')
    parser.add_argument('--channels_list', nargs='+', type=int, default=[32, 64, 128, 256, 320, 320],
        help="#channels of every levels of decoder in a top-down order")
    parser.add_argument('--kernel_size', type=int, default=3, help="size of conv kernels")
    parser.add_argument('--dropout_prob', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--norm', type=str, default='instance', 
        choices=['instance', 'batch', 'group'], help='type of norm')
    parser.add_argument('--num_classes', type=int, default=4, help='number of predicted classs')
    parser.add_argument('--weight_path', type=str, default=None, 
        help='path to pretrained encoder or decoder weight, None for train-from-scratch')
    parser.add_argument('--deep_supervision', action='store_true', 
        help='whether use deep supervision')
    parser.add_argument('--ds_layer', type=int, default=4, 
        help='last n layer to use deep supervision')

    # swin unetr
    parser.add_argument('--swin_feature_size', type=int, default=48)
    parser.add_argument('--swin_use_checkpoint',
        action='store_true', default=True,
        help='use gradient checkpointing in Swin UNETR (default: True). '
             'Pass --no_swin_use_checkpoint to disable.')
    parser.add_argument('--no_swin_use_checkpoint',
        dest='swin_use_checkpoint', action='store_false',
        help='disable gradient checkpointing in Swin UNETR')
    parser.add_argument('--swin_depths', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--swin_num_heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--swin_drop_rate', type=float, default=0.0)
    parser.add_argument('--swin_attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--swin_dropout_path_rate', type=float, default=0.0)

    # eval
    parser.add_argument('--save_model', action='store_true', default=False, 
        help='whether save model state')
    parser.add_argument('--save_pred', action='store_true', default=False, 
        help='whether save individual prediction')
    parser.add_argument('--eval_freq', type=int, default=10, help='eval frequency')
    parser.add_argument('--infer_batch_size', type=int, default=4, help='batchsize for inference')
    parser.add_argument('--patch_overlap', type=float, default=0.5, 
        help="overlap ratio between patches")
    parser.add_argument('--sw_batch_size', type=int, default=2, help="sliding window batch size")
    parser.add_argument('--sliding_window_mode', type=str, default='constant', 
        choices=['constant', 'gaussian'], help='sliding window importance map mode')

    args = parser.parse_args()


    if args.model == 'swin_unetr':
        model_tag = f"swin_fs{args.swin_feature_size}"
        if not args.swin_use_checkpoint:
            model_tag += "_nockpt"
    else:
        model_tag = f"{args.unet_arch}_{args.block}"

    exp_dir_name = [
        args.comment,
        args.dataset,
        model_tag,
        args.optim,
        args.scheduler,
        f"bs{args.batch_size}",
        f"ps{args.patch_size}",
        f"pos{args.pos_ratio}",
        f"neg{args.neg_ratio}",
    ]
    exp_dir_name.append(time.strftime("%m%d_%H%M%S", time.localtime()))
    exp_dir_name = "_".join([s for s in exp_dir_name if s])
    args.exp_dir = os.path.join(args.exp_dir, exp_dir_name)

    return args