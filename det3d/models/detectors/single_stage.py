from collections import OrderedDict
import torch
import torch.nn as nn

class SingleStageDetector(nn.Module):
    """Single Stage Detectors
    
    Args:
        reader: point cloud reader network
        backbone: 
        neck:
        head: detection head, eg. centerhead, transformerhead, ...
        post_processing:
    """
    def __init__(self,
                reader,
                backbone=None,
                neck=None,
                head=None,
                post_processing=None, 
                **kwargs):
        
        super(SingleStageDetector, self).__init__()
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.post_processing = post_processing

    def extract_feat(self, data):
        x = self.reader(data)
        if self.backbone is not None:
            x = self.backbone(*x)
        if self.neck is not None:
            x = self.neck(x)
        return x


    def forward(self, example):
        points = example['points']
        x = self.extract_feat(points)

        return self.head(x)

    def training_step(self, example):
        preds = self.forward(example)
        loss, log_vars = self.head.loss(example, preds)
        
        return loss, log_vars

    @torch.no_grad()
    def validation_step(self, example):
        preds = self.forward(example)
        outputs = self.head.predict(example, preds, self.post_processing)
        detections = {}
        for output in outputs:
            token = output["token"]
            for k, v in output.items():
                if k != "token":
                    output[k] = v.to(torch.device("cpu"))
            
            detections.update({token: output})
        return detections
    
    def get_downsample_factor(model_config):
        neck_cfg = model_config["neck"]
        downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
        if len(neck_cfg.get("us_layer_strides", [])) > 0:
            downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]
        backbone_cfg = model_config["backbone"]
        if backbone_cfg is not None:
            downsample_factor *= backbone_cfg["ds_factor"]
        downsample_factor = int(downsample_factor)
        assert downsample_factor > 0
        return downsample_factor
