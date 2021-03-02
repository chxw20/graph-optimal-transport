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
                name.text = str(type_map[ent["type"]])
                bndbox = ET.SubElement(obj, "bndbox")
                xmin, ymin, xmax, ymax = ET.SubElement(bndbox, "xmin"), ET.SubElement(bndbox, "ymin"), ET.SubElement(bndbox, "xmax"), ET.SubElement(bndbox, "ymax")
                xmin.text, ymin.text, xmax.text, ymax.text = [str(x) for x in ent["bbox"]]
            xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent='\t')
            open(f"data/{task}/annotations/{img_id}.xml", 'w').write(xmlstr)


if __name__ == "__main__":

    args = parse_args()
    gen_xml(args.task, os.listdir(f"data/{args.task}/json"))
