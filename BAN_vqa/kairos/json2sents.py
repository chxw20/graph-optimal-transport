import json
import sys
import os

from collections import defaultdict

def process(fpath):

    flickr_ent_type_map = {'people':0,'clothing':1,'bodyparts':2,'animals':3,'vehicles':4,'instruments':5,'scene':6,'other':7}
    with open("data/kairos_flickr_ent_mapping.txt", 'r') as f:
        kairos_to_flickr_ent_map = defaultdict(lambda: "other")
        for line in f:
            line = line.strip().split('\t')
            kairos_to_flickr_ent_map[line[0]] = line[1]

    data = json.load(open(fpath, 'r'))
    sents = data["sentences"]
    ents = data["entities"]

    results = []
    for sent in sents:
        ent = [e for e in ents if e["id"].startswith(sent["id"] + '-')]
        if len(ent) == 0:
            continue
        offset = sent["offset"]
        count = 1
        new_sent = sent["text"]
        for e in ent:
            if e["text"] == '':
                continue
            l = e["offset"] - offset
            r = l + e["length"]
            new_sent = new_sent[:l] + f"[/EN#{count}/{kairos_to_flickr_ent_map[e['type']]} " + new_sent[l:r] + "]" + new_sent[r:]
            offset -= (8 + len(str(count)) + len(kairos_to_flickr_ent_map[e['type']]))
            count += 1
        results.append(new_sent)

    return results


if __name__ == "__main__":

    dataset = sys.argv[1]
    fnames = os.listdir(f"data/{dataset}/json_output/primitives")
    if not os.path.exists(f"data/{dataset}/json_output/ent_sents"):
        os.makedirs(f"data/{dataset}/json_output/ent_sents")
    for fname in fnames:
        sents = process(f"data/{dataset}/json_output/primitives/{fname}")
        with open(f"data/{dataset}/json_output/ent_sents/{fname.replace('json', 'txt')}", 'w') as f:
            for line in sents:
                f.write(line + '\n')
