import os
from argparse import ArgumentParser
import model as md
from utils import create_link
import test as tst


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=100)   # A20551677 Changed from 200 to 100 (In lieu of time & resources available)
    parser.add_argument('--decay_epoch', type=int, default=50)  # A20551677 Changed from 100 to 10 (Learning rate linearly reduced to 0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--load_height', type=int, default=286)   
    parser.add_argument('--load_width', type=int, default=286)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=128)   # Ruthwik Adapala Changed this from 256 to 128 (In lieu of time & resources available)
    parser.add_argument('--crop_width', type=int, default=128)    # Ruthwik Adapala changed this from 256 to 128 (In lieu of time & resources available)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=1)    # A20551677 changed this to 1 to match the research paper.
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset_dir', type=str, default='/home/adapala/cs512/data/vangogh2photo')     # Ruthwik Adapala changed the directory location to load training and testing data.
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/vangogh2photo')              # Ruthwik Adapala changed the checkpoint directory name to vangogh2photo
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='dpn_gen')   # Ruthwik Adapala changed generator to dpn-gen (which is created)
    parser.add_argument('--dis_net', type=str, default='n_layers')
    args = parser.parse_args()
    return args


def main():
  args = get_args()

  create_link(args.dataset_dir)

  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      args.gpu_ids.append(id)
  print(not args.no_dropout)
  if args.training:
      print("Training")
      model = md.cycleGAN(args)
      model.train(args)
  if args.testing:
      print("Testing")
      tst.test(args)


if __name__ == '__main__':
    main()