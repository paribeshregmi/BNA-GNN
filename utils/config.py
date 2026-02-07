import os
import argparse

my_env = os.environ

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU id.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora','citeseer','pubmed'])
parser.add_argument('--fs', default=False, help='Set True for full supervised learning. Default: False (Semi-supervised)')

##args for BBGCN
parser.add_argument('--truncation', type=int, default=10,
                    help="truncation level for Beta Bernoulli process")
parser.add_argument("--a_prior", type=float, default=5.0,
                    help="a parameter for Beta distribution")
parser.add_argument("--b_prior", type=float, default=2.0,
                    help="b parameter for Beta distribution")
parser.add_argument("--num_samples", type=int, default=5,
                    help="Number of samples of Z matrix")
parser.add_argument("--temp", type=float, default=.1,
                    help="Temperature for posterior Concrete Bernoulli")
parser.add_argument('--kld_weight', type=float, default=1.0)
parser.add_argument('--arch_lr', type=float, default=0.1,
                    help='Architecture learning rate.')

parser.add_argument('--batch_size', type=int, default=102400)
parser.add_argument('--neighbor_order', type=int, default=1024, help='no. of neighbors to sample from each hop.')
parser.add_argument('--hops', type=int, default=8, help='no. of hops')


def get_args():
    args = parser.parse_args()
    fs_flag = '_fs' if args.fs else ''
    file_str = f'bb-gcn_{args.dataset}{fs_flag}_{args.batch_size}x{args.neighbor_order}x{args.hops}_trunc_{args.truncation}_hid_' \
               f'{args.hidden}_samples_{args.num_samples}_lr_{args.lr}_prior_{args.a_prior}x{args.b_prior}_klw_{args.kld_weight}_seed_{args.seed}'

    args.model_file = f'saved_models/{args.dataset}/{file_str}.pt'
    args.log_file = f'saved_logs/{args.dataset}/{file_str}.txt'
    print(args.model_file)
    return args