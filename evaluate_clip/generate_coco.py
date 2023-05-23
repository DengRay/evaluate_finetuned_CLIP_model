import json
import shutil
import os
from PIL import Image,ImageOps


file_path = "/home/dengyiru/fashion_attrs/coco_val_upper_17.json"
c = 0 
c2 = 0
count = 0
#tttt = 0
sour_path = "/home/dengyiru/fashion_attrs/upper/"
save_path = "/home/dengyiru/zero_shot/COCO_resize/"

with open(file_path) as f:
    data = json.load(f)

    img_address = data['images']
    img_attribute = data['annotations']
    #print(len(img_address))
    for item in img_address:
        c += 1
        
        #if c<5:
        img_path = item["file_name"] 
        img_id = item["id"]
        #tttt = img_id
        #print(tttt)
        for t in img_attribute:
            
            if t["image_id"] == img_id:
                src_path = sour_path + img_path
                box = t["bbox"]
                x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
                try:
                    img = Image.open(src_path)
                    size = img.size
                    #print("图片像素大小为：", size)
                    crop_img = img.crop((x1, y1, x2, y2))
                    #print(type(crop_img))
                    width, height = 200, 200  # 指定缩小后的像素大小
                    resized_img = crop_img.resize((width, height), resample=Image.LANCZOS)
                    #print(type(resized_img))
                    border_size = 12  # 指定边缘填充的大小
                    fill_color = (255, 255, 255)
                    expanded_img = ImageOps.expand(resized_img, border=(border_size, border_size, border_size, border_size), fill=fill_color)
                    #print(type(expanded_img))
                    #size = expanded_img.size
                    #print("!!")
                    #print("图片像素大小为：", size)
                    att = t["attribute_ids"]
                    c2 += 1
                    for tt in att:
                        #try:
                        dir_path = save_path + '{}'.format(tt["attribute_id"]) + '/' + '00' + '{}'.format(tt["value_id"])
                        if not os.path.exists(dir_path):
                            # 如果目录不存在，则创建它
                            os.makedirs(dir_path,exist_ok=True)
                        dst_path = save_path + '{}'.format(tt["attribute_id"]) + '/' + '00' + '{}'.format(tt["value_id"]) + '/'+ f'{img_id}.jpg'
                        #shutil.copy(src_path, dst_path)
                        expanded_img.save(dst_path)
                except:
                    continue
        #else:
           #break

    #print(img_address[100]["id"])
    #print(img_attribute[100]["image_id"])
               #c += 1
    #print(c)
    #print(c2)

    #print(tttt)
