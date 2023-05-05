import uuid
import pickle
import numpy as np
import copy
from tqdm import tqdm
from functools import reduce
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import box_utils


class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()


CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}


def label_to_type(label):
    if label <= 1:
        return int(label) + 1
    else:
        return 4



results = pickle.load(open('waymo_infos_val.pkl','rb'))


objects = metrics_pb2.Objects()
for pred in tqdm(results):
    pred = results[pred]
    pred_boxes = pred['box3d_lidar']
    pred_label = pred['label_preds']
    pred_score = pred['scores']

    for i in range(pred_boxes.shape[0]):
        det = pred_boxes[i]
        o = metrics_pb2.Object()
        o.context_name = pred['token'].split('-')[0]
        o.frame_timestamp_micros = int(pred['token'].split('-')[1])
        box = label_pb2.Label.Box()
        box.center_x = det[0]
        box.center_y = det[1]
        box.center_z = det[2]
        box.length = det[3]
        box.width = det[4]
        box.height = det[5]
        box.heading = det[-1]
        o.object.box.CopyFrom(box)
        o.score = pred_score[i]
        o.object.type = label_to_type(pred_label[i])
        objects.objects.append(o)

f = open('waymo_preds.bin', 'wb')
f.write(objects.SerializeToString())
f.close()