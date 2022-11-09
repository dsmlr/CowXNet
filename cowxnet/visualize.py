import cv2
import matplotlib.pyplot as plt
import cowxnet

alpha = 0.5

def visualize_cowxnet(img_path, boxes, visualizer=['cow_detect', 'cow_keypoint'], prediction_visualizer=[], focused_cow=[]):

    original_img = cv2.imread(img_path)
    img = original_img.copy()

    H, W, _ = img.shape

    for i in range(len(boxes)):
        
        try:
            cow_id = boxes[i]['cow']
        except:
            cow_id = 'x'
        
        if 'cow_detect' in visualizer:
            xmin, ymin, xmax, ymax = boxes[i]['box']
            start_point = (int(xmin), int(ymax))
            end_point = (int(xmax), int(ymin))

            if cow_id in focused_cow:
                color = (255, 255, 0, 1) # BGRA
            else:
#                 color = (2, 215, 255, 1) # Gold
#                 color = (0, 255, 0, 1) # Green
                color = (0, 0, 255, 1) # Red
            cv2.rectangle(img, start_point, end_point, color, 8)
        
        if 'cow_heat' in visualizer:
            xmin, ymin, xmax, ymax = boxes[i]['box']
            text = 'cow: {}, status: {}'.format(boxes[i]['cow'], boxes[i]['heat_label'])
            pos_x = int(xmax) - len(text) * 15
            pos_y = int(ymin) - 15

            if pos_x < 0:
                pos_x = 0
            elif pos_x > W:
                pos_x = int(xmin * W)

            if pos_y < 0:
                pos_y = int(ymax * H)

            text_position = (pos_x, pos_y)
            
            if boxes[i]['heat_label'] == 1:
                text_color = (0, 0, 255) # Red
            else:
                text_color = (254, 255, 0, 1) # Green

            cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        
        if 'cow_keypoint' in visualizer:
            for j in range(len(boxes[i]['keypoints'])):
                x = int(boxes[i]['keypoints'][j][0])
                y = int(boxes[i]['keypoints'][j][1])
                if x == 0 or y == 0:
                    continue

                color = (255, 0, 255, 1) # RGBA
                color = (255, 0, 255, 1)
                cv2.circle(img, (x, y), 15, color, -1)
                
        if 'cow_keypoint' in prediction_visualizer:
            for j in range(len(boxes[i]['predicted_keypoints'])):
                x = int(boxes[i]['predicted_keypoints'][j][0])
                y = int(boxes[i]['predicted_keypoints'][j][1])
                if x == 0 or y == 0:
                    continue

                xmin, ymin, xmax, ymax = boxes[i]['box']

                x, y = cowxnet.cow_keypoint.preprocess.get_new_coordinates(x, y, xmin, ymin, xmax, ymax, revert=True)

                color = (0, 0, 255)
                cv2.circle(img, (int(x), int(y)), 10, color, -1)

        if 'cow_detect' in prediction_visualizer:
            xmin, ymin, xmax, ymax = boxes[i]['predicted_box']
            start_point = (int(xmin), int(ymax))
            end_point = (int(xmax), int(ymin))
            color = (0, 0, 255, 1) # Red
            cv2.rectangle(img, start_point, end_point, color, 8)

    img = cv2.addWeighted(img, alpha, original_img, 1 - alpha, 0)

    return img