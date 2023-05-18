import torch
import numpy as np
import cv2
from time import time
from flask import Flask, render_template, Response

#egitilmis dosya yolu
model_name="../best.pt"
capture_index=0


"""
hangi kamerayı kullancağımız, hangi modeli kullanacağımız ekran kartı mı yoksa işlemci mi kullanacağız
ve bazı değişkenlere atama yapıyoruz
"""


def get_video_capture():
    """
    kameradan görüntü alıyoruz
    """

    return cv2.VideoCapture(capture_index) #cv2.VideoCapture(capture2_index)


def score_frame(frame):
    """
    kameradan aldığı görüntüyü modele sokarak ondan tahmin oranı alıyoruz
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def class_to_label(x):
    """
    classlarımızı labela dönüştürüyoruz.
    """
    return classes[int(x)]

def plot_boxes(results, frame):
    """
    aranan objenin hangi konumlar içinde olduğunu buluyoruz.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    sayac=0
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

model = torch.hub.load('C:\Users\sevva\Desktop\yolov5-master\yolov5-master', 'custom', path=model_name, force_reload=True,source="local")
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)



cap = get_video_capture()
assert cap.isOpened()


while True:

    ret, frame = cap.read()
    assert ret

    start_time = time()
    results = score_frame(frame)
    frame = plot_boxes(results, frame)

    end_time = time()
    fps = 1/np.round(end_time - start_time, 2)

    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('YOLOv5 Detection1', frame)
    out.write(frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
