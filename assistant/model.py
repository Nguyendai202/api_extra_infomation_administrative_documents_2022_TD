import os, cv2, numpy as np, torch
from assistant.create import Create
class Model:
    def __init__(self) -> None:
        cr = Create()
        self.opt = {
            "weights": "./yolov7-main/weights/best.pt", 
            "img-size": 640, 
            "conf-thres": 0.25, 
            "iou-thres" : 0.45, 
            "device" : 'cpu',  
            "classes" : cr.classes
        }
        self.yolo = cr.create_model_yolo(self.opt)
        self.vietocr = cr.create_model_vietocr()
def create_number(number):
        if number < 10:
            s = "0" + str(number)
            return s
        return str(number)
def delete_space(path : str):
        try:
            image = cv2.imread(path)
            image0 = image.copy()
            image0 = np.array(image0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh_image = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)[1]
            width = image.shape[1]
            height = image.shape[0]
            pixel_of_columns = []
            for w in range(0, width):
                s = 0
                for h in range(0, height):
                    s += (255 - thresh_image[h][w])
                pixel_of_columns.append(s)
            position = []
            for i in range(0, len(pixel_of_columns)):
                if pixel_of_columns[i] != 0:
                    position.append(i)
                    break
            for i in range(0, len(pixel_of_columns)):
                if pixel_of_columns[len(pixel_of_columns) - i - 1] != 0:
                    end = len(pixel_of_columns) - i - 1
                    position.append(end)
                    break
            part = image0[0 : height, position[0] : position[1] + 1]
            os.remove(path)
            cv2.imwrite(path, part)
        except FileNotFoundError:
            print("{0} not found!".format(path))
        except IndexError:
            print("List index out of range")
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        try:    
            # Resize and pad image while meeting stride-multiple constraints
            shape = img.shape[:2]  # current shape [height, width]
            if isinstance(new_shape, int):
                new_shape = (new_shape, new_shape)

            # Scale ratio (new / old)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            if not scaleup:  # only scale down, do not scale up (for better test mAP)
                r = min(r, 1.0)

            # Compute padding
            ratio = r, r  # width, height ratios
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
            if auto:  # minimum rectangle
                dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
            elif scaleFill:  # stretch
                dw, dh = 0.0, 0.0
                new_unpad = (new_shape[1], new_shape[0])
                ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

            dw /= 2  # divide padding into 2 sides
            dh /= 2

            if shape[::-1] != new_unpad:  # resize
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
            return img, ratio, (dw, dh)
        except ZeroDivisionError:
            print("{0} not found".format(img))
def replace_words(sentence : str):
        try:
            if sentence.startswith(" "):
                sentence = "-" + sentence
            words = sentence.split(' ')
            other_words= []
            for word in words:
                if len(word) == 0:
                    words.remove(word)
            chars = [",", "."]
            for word in words:
                if word[-1] in chars:
                    new_word = word[0 : len(word) - 1]
                    other_words.append(new_word)
                    other_words.append(word[len(word) - 1])   
                else: 
                    other_words.append(word)
            words_set = set(other_words)
            sample_words = list(words_set)
            words_dict = {}
            for word in sample_words:
                words_dict[word] = 0
            new_set = []
            for word in other_words:
                if word in sample_words and words_dict[word] == 0:
                    new_set.append(word)
                    words_dict[word] += 1
            out_sentence = ""
            for word in new_set:
                if word == "," or word == ".":
                    out_sentence += word
                else: 
                    out_sentence += " " + word
            return out_sentence
        except ArithmeticError:
            print("an error in {0}".format(sentence))
def split_rows(img_path):
        try:
            points = []
            other_points = []
            img0 = cv2.imread(img_path)
            img = img0.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            thresh_image = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)[1]
            height = img.shape[0]
            width = img.shape[1]  
            image_list = []
            value_y = []
            index_y = []
            split_y_const = 5
            split_index_y = []
            for h in range(0, height):
                row = []
                sum = 0
                for w in range(0, width):
                    row.append(thresh_image[h][w])
                    sum += thresh_image[h][w]
                value = float(sum/len(row))
                value_y.append([h, value])
        ###############
            for y in value_y:
                if y[1] < 253.0:
                    top = y[0]
                    break
            back_indx = len(value_y)
            while(back_indx >= 0):
                if value_y[back_indx - 1][1] <253.0:
                    bottom = value_y[back_indx - 1][0]
                    break
                else: 
                    back_indx -= 1
            for y in value_y:
                if y[1] >= 245.0:
                    index_y.append(y[0])
            for i in range(0, len(index_y) - 1):
                if index_y[i + 1] - index_y[i] >= split_y_const:
                    split_index_y.append([index_y[i], index_y[i + 1]])
        ###########################
            if len(split_index_y) > 1:
                steps = []
                for i in range(1, len(split_index_y)):
                    value = int((split_index_y[i - 1][1] + split_index_y[i][0]) / 2)
                    steps.append(value)
                left = split_index_y[0][0]
            ###########################
                points.append(top)
                for i in range(0, len(steps)):
                    points.append(steps[i])
                points.append(bottom)
                for i in range(0, len(points) - 2):
                    rate = (points[i + 2] - points[i + 1]) / (points[i + 1] - points[i])
                    if rate >= 1.6:
                        points[i + 1] = -1
                for point in points:
                    if point >= 0:
                        other_points.append(point)
                for i in range(0, len(other_points) - 1):
                    image_list.append(img0[other_points[i] : other_points[i + 1]])
            else:
                image_list.append(img0)
            return image_list
        except FileNotFoundError:
            print("{0} not found".format(img_path))
def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y      
def box_iou(box1, box2):
        def box_area(box):
        # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
