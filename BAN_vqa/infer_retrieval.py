"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, KairosFeatureDataset
import base_model
import utils
import pdb
from tqdm import tqdm

GPUID = 0
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='flickr30k', help='task name')
    parser.add_argument('--dset', type=str, default='scenario_data', help='dataset name')
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--epoch', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args


@torch.no_grad()
def infer(model, dataloader):
    upper_bound = 0
    N = 0
    logits_all = []
    a_all = []
    for i, (v, b, p, e, n, a, idx, types) in tqdm(enumerate(dataloader)):
        v = v.cuda()
        b = b.cuda()
        p = p.cuda()
        e = e.cuda()
        a = a.cuda()
        _, logits, gw = model(v, b, p, e, None)
        pdb.set_trace()
        n_obj = logits.size(2)
        logits.squeeze_()

        merged_logits = torch.cat(tuple(logits[j, :, :n[j][0]] for j in range(n.size(0))), -1).permute(1, 0)
        merged_a = torch.cat(tuple(a[j, :n[j][0], :n_obj] for j in range(n.size(0))), 0)

        logits_all.append(merged_logits)
        a_all.append(merged_a)

        N += n.sum().float()
        upper_bound += merged_a.max(-1, False)[0].sum().item()

    upper_bound = upper_bound / N

    pdb.set_trace()

    return upper_bound, torch.cat(logits_all, dim=0), torch.cat(a_all, dim=0)




if __name__ == '__main__':

    print('Infer using a given model optimized by training split using test split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dict_path = f'data/{args.task}/dictionary.pkl'
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset = KairosFeatureDataset('infer', dictionary, args.dset)
    
    args.op = ''
    args.gamma = 1

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma, args.task).cuda()
    model_data = torch.load(args.input+'/model'+('_epoch%d' % args.epoch if 0 < args.epoch else '')+'.pth')

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    model.train(False)

    pdb.set_trace()

    bound, logits_all, _ = infer(model, eval_loader)
    print('\tupper bound: %.2f' % (100 * bound))

    np.save(logits_all.detach().cpu().numpy(), f"data/{args.task}/results.npy")