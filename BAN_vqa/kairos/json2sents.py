import json
import sys
import os

from collections import defaultdict

def process(fpath, name=1, doc_entid_map={}):

    flickr_ent_type_map = {'people':0,'clothing':1,'bodyparts':2,'animals':3,'vehicles':4,'instruments':5,'scene':6,'other':7}
    with open("data/kairos_flickr_ent_mapping.txt", 'r') as f:
        kairos_to_flickr_ent_map = defaultdict(lambda: "other")
        for line in f:
            line = line.strip().split('\t')
            kairos_to_flickr_ent_map[line[0]] = line[1]

    data = json.load(open(fpath, 'r'))
    sents = data["sentences"]
    ents = data["entities"]

    corefs = [rel for rel in data["relations"] if "coreference" in rel["type"]]
    coref2name = {}

    results = []
    # doc_entid_map = dict()
    for sent in sents:
        ent = [e for e in ents if e["id"].startswith(sent["id"] + '-')]
        if len(ent) == 0:
            continue
        offset = sent["offset"]
        new_sent = sent["text"]
        new_ent = []
        for i in range(len(ent)):
            # find overlap with current entities for the current sentence
            overlap = False
            for j in range(len(new_ent)):
                if not (ent[i]["offset"] + ent[i]["length"] < new_ent[j]["offset"] or ent[i]["offset"] > new_ent[j]["offset"] + new_ent[j]["length"]):
                    overlap = True
                    # j stores the overlap entity
                    break
            if overlap:
                # print(ent[i]["text"], " ======= ", new_ent[j]["text"])
                if ent[i]["length"] >= new_ent[j]["length"]:
                    # new_ent[j] = ent[i]
                    # discard ent[i]
                    continue
            # find coreference
            if ent[i]["id"] in coref2name:
                ent[i]["ent_name"] = coref2name[ent[i]["id"]]
            else:
                for rel in corefs:
                    if ent[i]["id"] in rel["args"]:
                        for arg in rel["args"]:
                            coref2name[arg] = name
                        ent[i]["ent_name"] = name
                        name += 1
                        break
            # add entity
            if ent[i]["id"] not in coref2name:
                ent[i]["ent_name"] = name
                name += 1

            if overlap:
                new_ent[j] = ent[i]
            else:
                new_ent.append(ent[i])

        new_ent = sorted(new_ent, key=lambda e: e["offset"])
        
        for e in new_ent:
            if e["text"] == '':
                continue
            l = e["offset"] - offset
            r = l + e["length"]
            new_sent = new_sent[:l] + f"[/EN#{e['ent_name']}/{kairos_to_flickr_ent_map[e['type']]} " + new_sent[l:r] + "]" + new_sent[r:]
            offset -= (8 + len(str(e["ent_name"])) + len(kairos_to_flickr_ent_map[e["type"]]))
            doc_entid_map[e["ent_name"]] = e["id"]
        results.append(new_sent)

    return results, name, doc_entid_map




if __name__ == "__main__":

    dataset = sys.argv[1]
    fnames = os.listdir(f"data/{dataset}/json_output/primitives")
    if not os.path.exists(f"data/{dataset}/json_output/ent_sents"):
        os.makedirs(f"data/{dataset}/json_output/ent_sents")

    with open(f"data/{dataset}/topic_mapping.txt", 'r') as f:
        topic2docs = defaultdict(list)
        for line in f:
            line = line.strip().split('\t')
            topic2docs[line[1]].append(line[0])

    for topic in topic2docs:
        print(f"processing topic {topic} ...")
        name = 1
        doc_entid_map = dict()
        topic_fnames = [f"{doc}.json" for doc in topic2docs[topic] if f"{doc}.json" in fnames]
        if topic[-4:] == "_spa":
            assert topic_fnames == []
            continue
        for fname in topic_fnames:
            sents, name, doc_entid_map = process(f"data/{dataset}/json_output/primitives/{fname}", name, doc_entid_map)
            with open(f"data/{dataset}/json_output/ent_sents/{fname.replace('json', 'txt')}", 'w') as f:
                for line in sents:
                    f.write(line + '\n')
        with open(f"data/{dataset}/json_output/{topic.replace('_eng', '')}.json", 'w') as f:
            json.dump(doc_entid_map, f)
