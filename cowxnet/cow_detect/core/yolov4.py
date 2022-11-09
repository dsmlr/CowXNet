import os
import json

def load_anno(file_path):

    with open(file_path) as json_file:
        json_obj = json.load(json_file)
    
    objs = json_obj['shapes']
    meta = {
        'objects': objs,
        'image_size': {
            'width': json_obj['imageWidth'],
            'height': json_obj['imageHeight']
        }
    }
    return meta

def convert_anno_to_labelmg_format(file_path):
    
    meta = load_anno(file_path)
    
    file_name = file_path.split('/')[-1].split('.')[0]
    path_to_file = file_path.split(file_name)[0]
    
    w = meta['image_size']['width']
    h = meta['image_size']['height']
    
    f = open(os.path.join(path_to_file, 'yolo_anno', file_name + '.txt'), 'w')
    for obj in meta['objects']:
        
#         _class = int(obj['label'])
        _class = 0
        
        (_xmin, _ymin), (_xmax, _ymax) = obj['points']

        xmin = min(_xmin, _xmax)
        xmax = max(_xmin, _xmax)
        ymin = min(_ymin, _ymax)
        ymax = max(_ymin, _ymax)

        ymin = max(0, ymin)
        xmin = max(0, xmin)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h

        obj_w = (xmax - xmin) / w
        obj_h = (ymax - ymin) / h
        
        f.write('{} {} {} {} {}\n'.format(_class, x_center, y_center, obj_w, obj_h))
        
    f.close()