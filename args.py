import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument("--data_dir", default="/home/user/lyx/Programs/DocRED", type=str)
parser.add_argument("--transformer_type", default="bert", type=str)
parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

parser.add_argument("--train_file", default="train_distant1.json", type=str)
parser.add_argument("--dev_file", default="dev.json", type=str)
parser.add_argument("--test_file", default="test.json", type=str)

parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--max_seq_length", default=1024, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--exp-dir', default='experiment/MoPro_V1', type=str,
                    help='experiment directory')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',choices=['resnet50',])
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[40, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:5678', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--num-class', default=97, type=int)
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.999, type=float,
                    help='momentum for computing the momving average of prototypes')

parser.add_argument('--temperature', default=0.1, type=float,
                    help='contrastive temperature')

parser.add_argument('--w-inst', default=1, type=float,
                    help='weight for instance contrastive loss')
parser.add_argument('--w-proto', default=1, type=float,
                    help='weight for prototype contrastive loss')

parser.add_argument('--start_clean_epoch', default=11, type=int,
                    help='epoch to start noise cleaning')
parser.add_argument('--pseudo_th', default=0.8, type=float,
                    help='threshold for pseudo labels')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='weight to combine model prediction and prototype prediction')