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
from collections import defaultdict

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
    results = defaultdict(list)
    # doc_ent_id: [(img_id, box_id, match_prob)]
    for i, (v, b, p, e, n, idx, types, img_ids) in tqdm(enumerate(dataloader)):
        v = v.cuda()
        b = b.cuda()
        p = p.cuda()
        e = e.cuda()

        _, logits, gw = model(v, b, p, e, None)
        # pdb.set_trace()
        n_obj = logits.size(2)
        logits.squeeze_()

        merged_logits = torch.cat(tuple(logits[j, :, :n[j][0]] for j in range(n.size(0))), -1).permute(1, 0)

        prob, inds = merged_logits.softmax(dim=1).max(dim=1)
        prob, inds = prob.detach().cpu().numpy(), inds.detach().cpu().numpy()

        doc_ent_ids = idx[idx != 0]
        img_ids = img_ids.view(-1).detach().cpu().numpy()
        for doc_ent_id, img_id, box_id, match_prob in zip(doc_ent_ids, img_ids, inds, prob):
            results[doc_ent_id].append((img_id, box_id, match_prob))


    pdb.set_trace()

    return results




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

    results = infer(model, eval_loader)

    np.save(results, f"data/{args.task}/results.npy")