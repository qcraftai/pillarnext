import numpy as np
import det3d.core.bbox.box_np_ops as box_np_ops

class Flip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
        assert 0 <= flip_prob[0] < 1
        assert 0 <= flip_prob[1] < 1

    def __call__(self, res):
        # x flip
        if self.flip_prob[0] > 0:
            enable = np.random.choice([False, True], replace=False, 
                                        p=[1 - self.flip_prob[0], self.flip_prob[0]])
            if enable:
                res['points'][:, 1] = -res['points'][:, 1]
                if 'annotations' in res and 'gt_boxes' in res['annotations']:
                    mask = np.isnan(res['annotations']['gt_boxes'])
                    res['annotations']['gt_boxes'][mask] = 0
                    res['annotations']['gt_boxes'] = box_np_ops.flip(res['annotations']['gt_boxes'], axis='x') 
                    res['annotations']['gt_boxes'][mask] = np.nan
        # y flip
        if self.flip_prob[1] > 0:
            enable = np.random.choice(
                [False, True], replace=False, p=[1 - self.flip_prob[1], self.flip_prob[1]])
            if enable:
                res['points'][:, 0] = -res['points'][:, 0]
                
                if 'annotations' in res and 'gt_boxes' in res['annotations']:
                    mask = np.isnan(res['annotations']['gt_boxes'])
                    res['annotations']['gt_boxes'][mask] = 0
                    res['annotations']['gt_boxes'] = box_np_ops.flip(res['annotations']['gt_boxes'], axis='y') 
                    res['annotations']['gt_boxes'][mask] = np.nan
        return res

class Scaling(object):
    def __init__(self, scale):
        self.min_scale, self.max_scale = scale
       
    def __call__(self, res):
        noise_scale =  np.random.uniform(self.min_scale, self.max_scale)
        res['points'][:, :3] *= noise_scale
        if 'annotations' in res and 'gt_boxes' in res['annotations']:
            mask = np.isnan(res['annotations']['gt_boxes'])
            res['annotations']['gt_boxes'][mask] = 0
            res['annotations']['gt_boxes'] = box_np_ops.scaling(res['annotations']['gt_boxes'], noise_scale)
            res['annotations']['gt_boxes'][mask] = np.nan
        return res



class Rotation(object):
    def __init__(self, rotation):
        self.rotation = rotation
    
    def __call__(self, res):
        noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        
        res['points'][:, :3] = box_np_ops.yaw_rotation(res['points'][:, :3], noise_rotation)

        if 'annotations' in res and 'gt_boxes' in res['annotations']:
            mask = np.isnan(res['annotations']['gt_boxes'])
            res['annotations']['gt_boxes'][mask] = 0
            res['annotations']['gt_boxes'] = box_np_ops.rotate(res['annotations']['gt_boxes'], noise_rotation)
            res['annotations']['gt_boxes'][mask] = np.nan
        return res

class Translation(object):
    def __init__(self, noise):
        self.noise = noise 
    
    def __call__(self, res):
        noise_translate = np.random.normal(0, self.noise, 1)
        res['points'][:, :3] += noise_translate
        if 'annotations' in res and 'gt_boxes' in res['annotations']:
            mask = np.isnan(res['annotations']['gt_boxes'])
            res['annotations']['gt_boxes'][mask] = 0
            res['annotations']['gt_boxes'] = box_np_ops.translate(res['annotations']['gt_boxes'], noise_translate)
            res['annotations']['gt_boxes'][mask] = np.nan
        return res
