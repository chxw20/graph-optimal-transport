import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='scenario_data', help='dataset name')
    args = parser.parse_args()
    return args


def gen_xml(task, fnames):

    # kairos_ent_types = ["ABS", "AML", "BAL", "BOD", "COM", "FAC", "GPE", "INF", "LAW",
    #                     "LOC", "MHI", "MON", "NAT", "ORG", "PER", "PLA", "PTH", "RES",
    #                     "SEN", "SID", "TTL", "VAL", "VEH", "WEA"]
    flickr_ent_type_map = {'people':0,'clothing':1,'bodyparts':2,'animals':3,'vehicles':4,'instruments':5,'scene':6,'other':7}
    with open("data/kairos_flickr_ent_mapping.txt", 'r') as f:
        kairos_to_flickr_ent_map = {}
        for line in f:
            line = line.strip().split('\t')
            kairos_to_flickr_ent_map[line[0]] = line[1]

    type_map = dict([(t, flickr_ent_type_map[kairos_to_flickr_ent_map[t]]) for t in kairos_to_flickr_ent_map])

    ent_idx = 1
    for fname in fnames:
        data = json.load(open(f"data/{task}/json/{fname}"))
        imgid2ents = defaultdict(list)
        for ent in data["entities"]:
            img_id = ent["id"].split('-')[1] + '-f' + str(int(ent["id"].split('-')[2].replace('f', '')) + 1)
            imgid2ents[img_id].append(ent)
        for img_id in imgid2ents:
            # xml_fname = img_id + ".xml"
            data = ET.Element("annotation")
            img_fname = ET.SubElement(data, "filename")
            img_fname.text = img_id + ".jpg"
            for ent in imgid2ents[img_id]:
                if ent["bbox"][0] == ent["bbox"][1]:
                    continue
                obj = ET.SubElement(data, "object")
                name = ET.SubElement(obj, "name")
                # name.text = str(type_map[ent["type"]])
                name.text = str(ent_idx)
                ent_idx += 1
                bndbox = ET.SubElement(obj, "bndbox")
                xmin, ymin, xmax, ymax = ET.SubElement(bndbox, "xmin"), ET.SubElement(bndbox, "ymin"), ET.SubElement(bndbox, "xmax"), ET.SubElement(bndbox, "ymax")
                xmin.text, ymin.text, xmax.text, ymax.text = [str(x) for x in ent["bbox"]]
            xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent='\t')
            open(f"data/{task}/annotations/{img_id}.xml", 'w').write(xmlstr)

    print(f"text start entity = {ent_idx}")


def create_topic_doc_map(task):

    topic2doc = defaultdict(list)
    primitives_list = os.listdir(f"data/{task}/json_output/primitives")
    primitives_list = [fname.replace(".json", '') for fname in primitives_list]
    with open(f"data/{task}/topic_mapping.txt", 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] in primitives_list:
                topic = '_'.join(line[1].split('_')[:-1])
                topic2doc[topic].append(line[0])

    topic2doc["test"] = topic2doc["Boston_Marathon_Bombing_April_2013"] # test

    with open(f"data/{task}/topic_doc_map.json", 'w') as f:
        json.dump(topic2doc, f)


if __name__ == "__main__":

    args = parse_args()
    gen_xml(args.task, os.listdir(f"data/{args.task}/json"))
    create_topic_doc_map(args.task)
