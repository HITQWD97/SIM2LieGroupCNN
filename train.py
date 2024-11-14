from trainer import Trainer
import torch
import argparse
import os
from torch.utils.data import DataLoader
from lie_groupnet import SIM2LieGroup
from dataset import MNISTRS
from utils import cosLr
from lie_group import SIM2
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='SIM2LieGroup')
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--result', default='./results')
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--group', default=SIM2())
    parser.add_argument('--chin', default=1, type=int)
    parser.add_argument('--data_path', default='./dataset/')
    args = parser.parse_args()
    if not os.path.exists(args.result):
        os.mkdir(args.result)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    split_datasets = {}
    split_datasets['train'] = MNISTRS(args.data_path)
    split_datasets['test'] = MNISTRS(args.data_path, train=False)
    model = SIM2LieGroup(args.chin, num_targets=split_datasets['train'].num_targets, group=args.group).to(device)
    dataloaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=(k=='train'), num_workers=0, pin_memory=False)
              for k, v in split_datasets.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_sched = cosLr(args.num_epoch)
    lr_schedulers = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)
    Trainer(model, dataloaders, optimizer, lr_schedulers, criterion, device, args)
if __name__ == '__main__':
    main()