from keras.models import load_model
import cv2 as openCV
import numpy as np

model = load_model('model-017.model')

faceClassifier = openCV.CascadeClassifier("haarcascade_frontalface_default.xml")

camera = openCV.VideoCapture(0)
labels = {0 : "SAFE", 1 : "NOT SAFE"}
colors = {0 : (0, 255, 0), 1 : (0, 0, 255)}


while True:
    ret, img = camera.read()
    gray = openCV.cvtColor(img, openCV.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y:y + w, x:x + w]
        resized = openCV.resize(face, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        openCV.rectangle(img, (x, y), (x + w, y + h), colors[label], 2)
        openCV.rectangle(img, (x, y - 40), (x + w, y), colors[label], -1)
        openCV.putText(
            img, labels[label],
            (x, y - 10),
            openCV.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    openCV.imshow('LIVE MASK DETECTION', img)
    key = openCV.waitKey(1)

    if (key == 27):
        break

openCV.destroyAllWindows()
camera.release()
