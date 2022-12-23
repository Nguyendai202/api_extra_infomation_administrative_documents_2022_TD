import sys
import torch
import numpy as np
from numpy import random
import torchvision

sys.path.append("./yolov7-main")
from utils.general import check_img_size, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from assistant.model import letterbox, xywh2xyxy, box_iou, Model
class YOLOv7:
    def __init__(self):
        model = Model()
        opt = model.opt
        self.conf = opt["conf-thres"]
        self.device = select_device(opt["device"])
        self.iou = opt["iou-thres"]
        self.classes = opt["classes"] 
        self.imgsize = opt["img-size"]
        self.model = model.yolo
        self.vietocr = model.vietocr
    def preprocessing(self, image):
        half = self.device.type != 'cpu'
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsize, s=stride)  # check img_size
        img = letterbox(image, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))
        return img
    def run(self, image):
        img0 = image.copy()
        im0 = image.copy()
        output_boxes_corner = []
        nb = 0
        device = self.device
        model = self.model
        half = device.type != 'cpu'
        if half:
            model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        img = self.preprocessing(img0)
        pred = model(img, augment= False)[0]
        classes = []
        for class_name in self.classes:
            classes.append(names.index(class_name))
        if classes:
            classes = [i for i in range(len(names)) if i not in classes]
        conf_thres = self.conf
        iou_thres = self.iou
        labels = ()
        multi_label = False
        agnostic = False
        nc = pred.shape[2] - 5  # number of classes
        xc = pred[..., 4] > conf_thres  # candidates
            # Settings
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]
        for xi, x in enumerate(pred):  # image index, image inference
                x = x[xc[xi]]
                if labels and len(labels[xi]):
                    l = labels[xi]
                    v = torch.zeros((len(l), nc + 5), device=x.device)
                    v[:, :4] = l[:, 1:5]  # box
                    v[:, 4] = 1.0  # conf
                    v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                    x = torch.cat((x, v), 0)

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                if nc == 1:
                    x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                        # so there is no need to multiplicate.
                else:
                    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = xywh2xyxy(x[:, :4])
                # Detections matrix nx6 (xyxy, conf, cls)
                if multi_label:
                    i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                    x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                else:  # best class only
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                n = x.shape[0]  # number of boxes
                if not n:  # no boxes
                    continue
                elif n > max_nms:  # excess boxes
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

                # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy
                output[xi] = x[i]
        pred = output
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    nb += 1
                    label = f'{names[int(cls)]}' #{conf:.2f}
                    output_boxes_corner.append([xywh, label])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        return output_boxes_corner