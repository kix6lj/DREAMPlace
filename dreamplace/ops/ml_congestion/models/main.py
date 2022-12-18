import os
import json
import torch
import argparse
import sys
import torch.optim as optim
from routenet import RouteNet
from build_dataset import build_dataset
from utils import MSELoss
from tqdm import tqdm

def checkpoint(model, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
    torch.save({'state_dict': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
        

def train(args):
    arg_dict = vars(args)

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)
    
    print('===> Building model')
    # Initialize model parameters
    print(type(arg_dict['in_channels']), type(arg_dict['out_channels']))
    model = RouteNet(arg_dict['in_channels'], arg_dict['out_channels'])
    model = model.cuda()
    
    # Build loss
    loss = MSELoss()

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000

    while iter_num < arg_dict['max_iters']:
        with tqdm(total=print_freq) as bar:
            for feature, label, _ in dataset:        
                input, target = feature.cuda(), label.cuda()

                prediction = model(input)

                optimizer.zero_grad()
                pixel_loss = loss(prediction, target)

                epoch_loss += pixel_loss.item()
                pixel_loss.backward()
                optimizer.step()

                iter_num += 1
                
                bar.update(1)

                if iter_num % print_freq == 0:
                    break

        print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'])
        epoch_loss = 0

def evaluate(args):
    arg_dict = vars(args)
    arg_dict['ann_file'] = arg_dict['ann_file_test']

    model = RouteNet(arg_dict['in_channels'], arg_dict['out_channels'])
    model.init_weights(arg_dict['pretrained'])
    model.cuda()
    dataset = build_dataset(arg_dict)

    iter_num = 0
    total_loss = 0
    print_freq = 100

    loss = MSELoss()

    with tqdm(total=len(dataset)) as bar:
        for feature, label, _ in dataset:
            input, target = feature.cuda(), label.cuda()
            with torch.no_grad():
                prediction = model(input)
                pixel_loss = loss(prediction, target)
                total_loss += pixel_loss.item()
            iter_num += 1
            bar.update(1)

    print(total_loss / iter_num)

    return

def switching(parser):
    args = parser.parse_args()
    if (args.test_mode == True):
        evaluate(args)
    else:
        train(args)
    return

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--ispd_dir', default=None)
    parser.add_argument('--dataroot', default=None)
    parser.add_argument('--ann_file_train', default=None)
    parser.add_argument('--ann_file_test', default=None)
    parser.add_argument('--max_iters', default=10000, type=int)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--save_path', default='work_dir/routenet')
    parser.add_argument('--aug_pipeline', default=['Flip'])
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=0, type=int)
    parser.add_argument('--test_mode', action='store_true')
    parser.set_defaults(test_mode=False)
    switching(parser)