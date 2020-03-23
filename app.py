from flask import Flask, render_template, Response, url_for , redirect
import cv2
import numpy as np 
import tensorflow as tf 
from Inference import Inference
from labelutil import labelParser
app = Flask(__name__)

def video_frame():
    rtsp_url ='rtsp://172.30.1.51:8554/mjpeg/1'
    inference =  Inference()
    model = inference.read_model()
    label = labelParser()
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            frame = cv2.flip(cv2.flip(frame,1),0)
            frame = tf.convert_to_tensor(tf.expand_dims(frame,axis=0))
            result = model(frame)
            frame = np.squeeze(frame,axis=0)
            inference.draw_bbox(frame,result,label)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(video_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




def on_click():
    print('Pressed Button')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)