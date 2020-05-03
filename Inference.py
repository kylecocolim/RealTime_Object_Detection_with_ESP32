import tensorflow as tf 
import numpy as np
import cv2 
from labelutil import labelParser


class Inference():
    def __init__(self,confidence_score=0.5):
        self.MODEL_PATH = './Model_config/Model/saved_model'
        self.LABEL_PATH = './Model_config/label_map.pbtxt'
        self.confidence_score = confidence_score
    def category_parserd(self,parsedlabel,idx):
        return parsedlabel[idx]
    def read_model(self):
        model = tf.saved_model.load(self.MODEL_PATH)
        model = model.signatures['serving_default']
        print('Model Loaded')
        return model
    def draw_bbox(self,img,result,parsedlabel):
        width , height , _ = img.shape 
        bboxes= result['detection_boxes'][0]
        class_names = result['detection_classes'][0]
        scores = result['detection_scores'][0]
        draw_boxes = []
        for idx in range(len(scores)):
            if float(scores[idx].numpy()) >=self.confidence_score:
                draw_boxes.append([str(int(class_names[idx].numpy())),[int(bboxes[idx][0]*width),int(bboxes[idx][1]*height),int(bboxes[idx][2]*width),int(bboxes[idx][3]*height)]])
            if len(draw_boxes) >= 1:
                for i in range(len(draw_boxes)):
                    class_name = self.category_parserd(parsedlabel,draw_boxes[i][0])
                    print(class_name)
                    ymin = draw_boxes[i][1][0]
                    xmin = draw_boxes[i][1][1]
                    ymax = draw_boxes[i][1][2]
                    xmax = draw_boxes[i][1][3]
                    img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color=(255,255,255),thickness=1)
                    img = cv2.putText(img,class_name,(xmin,ymin-10),color=(255,255,255),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0)
                    
            return img

    def expand_dims(self,img):
        return tf.convert_to_tensor(tf.expand_dims(img,axis=0))
     
        

if __name__ == "__main__":
    model = Inference().read_model()
    label = labelParser()
    img = cv2.imread('test.jpg')
    img = tf.expand_dims(img,axis=0)
    result = model(img)
    img = np.squeeze(img,axis=0)
    Inference().draw_bbox(img,result,label)
    cv2.imshow('img',img)
    cv2.waitKey(0)