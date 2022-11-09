import cv2
import numpy as np

def visualize_detections(img_path, boxes, type='crop'):

    """
        type:
            - crop_with_pad
            - crop
    """

    img = cv2.imread(img_path)

    height, width, _ = img.shape

    results = []
    
    for i in range(len(boxes)):

        xmin, ymin, xmax, ymax = boxes[i]['box']
        xmin, ymin, xmax, ymax = (xmin, ymin, xmax, ymax)
        w = int(xmax - xmin)
        h = int(ymax - ymin)

        crop_img = img[int(ymin):int(ymin) + h, int(xmin):int(xmax)]

        if type == 'crop':
            result_img = crop_img.copy()

        elif type == 'crop_with_pad':
            x_offset = int((width - crop_img.shape[1])/2)
            y_offset = int((height - crop_img.shape[0])/2)

            crop_img_with_pad = 255 * np.ones_like(np.ones((height, width, 3)))

            crop_img_with_pad[y_offset: y_offset + crop_img.shape[0], x_offset: x_offset + crop_img.shape[1]] = crop_img
            result_img = crop_img_with_pad.copy().astype(np.uint8)

        results.append(result_img)

    return results