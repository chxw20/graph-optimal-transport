import json
import sys
import os

def process(fpath):

    data = json.load(open(fpath, 'r'))
    sents = data["sentences"]
    ents = data["entities"]

    results = []
    for sent in sents:
        ent = [e for e in ents if e["id"].startswith(sent["id"] + '-')]
        offset = sent["offset"]
        count = 1
        new_sent = sent["text"]
        for e in ent:
            if e["text"] == '':
                continue
            l = e["offset"] - offset
            r = l + e["length"]
            new_sent = new_sent[:l] + f"[/EN#{count}{e['type']} " + new_sent[l:r] + "]" + new_sent[r:]
            offset -= (7 + len(str(count)) + len(e['type']))
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
