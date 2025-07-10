import pyarrow.parquet as pq
import json

src_file = "/teamspace/studios/this_studio/raw_res/LLaVA-OneVision-Data/train-00000-of-00005.parquet"
dst_imgs_dir = "/teamspace/studios/this_studio/raw_res/LLaVA-OneVision-Data/textcap/images"
dst_json_file = (
    "/teamspace/studios/this_studio/VideoLLaMA3/DATASETS/STAGE1/annotations.json"
)

table = pq.read_table(src_file)
print("table:", table.column_names)

name_pattern = "image_{}.jpg"
maxnum: int = 0
MAXNUM = -1  # any limit?
for idx, img_bytes in enumerate(table["image"]):
    maxnum += 1
    if -1 != MAXNUM and maxnum > MAXNUM:
        break
    # save image
    name = name_pattern.format(idx)
    with open(f"{dst_imgs_dir}/{name}", "wb") as f:
        f.write(img_bytes.as_py()["bytes"])

maxnum = 0
json_dic = []  # the dest json data
item = {}  # single data item
for idx, img_bytes in enumerate(table["conversations"]):
    maxnum += 1
    if -1 != MAXNUM and maxnum > MAXNUM:
        break
    name = name_pattern.format(idx)
    item["image"] = [f"{dst_imgs_dir}/{name}"]
    item["conversations"] = img_bytes.as_py()
    json_dic.append(item.copy())

# save json
with open(dst_json_file, "w") as d:
    json.dump(json_dic, d)
