import json
import shutil
import os

file_path = "/home/dengyiru/fashion_attrs/coco_val_upper_17.json"
c = 0 
c2 = 0

#tttt = 0
sour_path = "/home/dengyiru/fashion_attrs/upper/"
save_path = "/home/dengyiru/zero_shot/COCO_resize/"

with open(file_path) as f:
    data = json.load(f)

    img_att = data['attributes']

    for item in img_att:
        
        save_ = save_path + f"{c}.txt"
        specific = item["attribute_value"]
        for j in specific:
            st = j["value_name"]
            file = open(save_, "a")
            file.write(st+"\n")
            file.close()
        c+=1
