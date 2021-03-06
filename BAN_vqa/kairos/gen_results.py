import argparse
import os
import sys
sys.path.append("./")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json

from dataset import Dictionary, KairosFeatureDataset
import base_model
import utils
import pdb
from tqdm import tqdm
from collections import defaultdict

pdb.set_trace = lambda: None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="test", help="dataset name")
    parser.add_argument("--topic", type=str, default="test", help="topic name")
    parser.add_argument("--task", type=str, default="flickr30k", help="task name")
    parser.add_argument("--p_th", type=float, default=0.5, help="threshold for entity matching")
    parser.add_argument("--iou_th", type=float, default=0.5, help="threshold for iou")
    args = parser.parse_args()
    return args



def gen_coref(task, topic, fnames, results, dataset, p_th=0.5, iou_th=0.5):

    img_idx2id = pickle.load(open(f"data/{task}/infer_imgid2idx.pkl", "rb"))
    # img_idx2id = dict([(v, k) for (k, v) in img_id2idx.items()])

    imgid2ents = defaultdict(list)

    imgid2bboxes = defaultdict(list)
    imgid2entids = defaultdict(list)

    for fname in fnames:
        data = json.load(open(f"data/{task}/json/{fname}"))
        for ent in data["entities"]:
            img_idx = ent["id"].split('-')[1] + '-f' + str(int(ent["id"].split('-')[2].replace('f', '')) + 1)
            img_id = img_idx2id[img_idx]
            imgid2ents[img_id].append(ent)
    for img_id in imgid2ents:
        dst_bboxes = []
        dst_entids = []
        for ent in imgid2ents[img_id]:
            if ent["bbox"][0] == ent["bbox"][1]:
                continue
            dst_bboxes.append(ent["bbox"])
            dst_entids.append((ent["id"], ent["type"]))

        imgid2bboxes[img_id] = dst_bboxes
        imgid2entids[img_id] = dst_entids

    corefs = []
    doc_entid_map = json.load(open(f"data/{task}/json_output/{topic}.json"))
    for (doc_ent_id, res) in tqdm(results.items()):
        pdb.set_trace()
        if str(doc_ent_id) not in doc_entid_map:
            continue
        for (img_id, box_id, p) in res:
            pdb.set_trace()
            if p < p_th:
                continue
            try:
                feat_bbox = dataset.bbox[dataset.pos_boxes[img_id][0]:dataset.pos_boxes[img_id][1]][box_id]
            except:
                continue
            matched_bboxes = []
            for i, dst_bbox in enumerate(imgid2bboxes[img_id]):
                if utils.calculate_iou(feat_bbox, np.array(dst_bbox)) > iou_th:
                    matched_bboxes.append((i, iou_th))
            pdb.set_trace()
            if len(matched_bboxes) == 0:
                continue
            matched_bboxes = sorted(matched_bboxes, key=lambda x: x[1], reverse=True)
            if doc_entid_map[str(doc_ent_id)][1] == imgid2entids[img_id][matched_bboxes[0][0]][1]:
                corefs.append((doc_entid_map[str(doc_ent_id)][0], imgid2entids[img_id][matched_bboxes[0][0]][0], p))

    return corefs




if __name__ == "__main__":

    args = parse_args()

    dict_path = f"data/{args.task}/dictionary.pkl"
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset = KairosFeatureDataset("infer", dictionary, args.dset)

    # eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    results = np.load(f"data/{args.dset}/results.npy", allow_pickle=True).item()

    fnames = os.listdir(f"data/{args.dset}/json")
    corefs = gen_coref(args.dset, args.topic, fnames, results, eval_dset, args.p_th, args.iou_th)
    with open(f"data/{args.dset}/results.txt", 'w') as f:
        for (text_id, img_id, p) in corefs:
            f.write(f"{text_id}\t{img_id}\t{p:.4f}\n")