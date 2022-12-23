import os,  sys

sys.path.append("./yolov7-main")
from vietnameseocr.vietocr.tool.predictor import Predictor
from vietnameseocr.vietocr.tool.config import Cfg
from models.experimental import attempt_load
from utils.torch_utils import select_device

class Create:
    def __init__(self):
        self.fields = {
        'time' : 'Thời gian, địa điểm:',
        'heading' : 'Cơ quan:',
        'number' : 'Số hiệu:',
        'title_1' : 'Tiêu đề:',
        'title_2' : 'Tiêu đề:', 
        'signature' : 'Đóng dấu:', 
        'author' : 'Người ký:', 
        'to' : 'Nơi nhận:'
    }
        self.classes = ['author', 'heading', 'number', 'signature', 'time', 'title_1', 'title_2', 'to']
    
    def create_label_folder(self, labels_big_folder):
        classes = self.classes
        for i in classes:
            folder_path = os.path.join(labels_big_folder, i)
            os.mkdir(folder_path)
    def create_label_variable(self, labels_big_folder):
        labels_list = os.listdir(labels_big_folder)
        labels_list = sorted(labels_list)
        labels_variable = []
        for label in labels_list:
            labels_variable.append(os.path.join(labels_big_folder, label))
        return labels_variable
    def create_model_vietocr(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = "./yolov7-main/weights/transformerocr_version_2.pth"
        config['cnn']['pretrained']=False
        config['device'] = 'cpu'
        config['predictor']['beamsearch']=False
        detector = Predictor(config)
        return detector
    def create_model_yolo(self, opt_corner):
        weights = opt_corner['weights']
        device = select_device(opt_corner['device'])
        model = attempt_load(weights, map_location=device)  # load FP32 model
        return model
    
