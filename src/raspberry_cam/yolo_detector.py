import cv2
import numpy as np
import onnxruntime as ort
# 640/480
sess = None

def load_model(model_path):
    global sess
    sess = ort.InferenceSession(model_path)

def inference(image):
    # 0. convert BGR to RGB
    # 1. resize to 640x640, keep aspect ratio, pad with zeros
    # 2. normalize to [0, 1]
    # 3. transpose to (C, H, W)
    # 4. add batch dimension
    # 5. run inference
    image = image[..., [2, 1, 0]] #BGR->RGB
    back = np.zeros((640, 640, 3), np.uint8)

    H, W, _ = image.shape
    if H > W:
        h = 640
        w = W * 640 // H
    else:
        w = 640
        h = H * 640 // W

    resized = cv2.resize(image, (w, h))
    back[0:h, 0:w] = resized

    input = back.astype(np.float32) / 255
    input = input.transpose(2, 0, 1)[None]

    res = sess.run(None, input_feed={'images':input})[0][0]
    return res

def draw_boxes(image, res):
    H, W, _ = image.shape
    long = max(H, W)
    rate = long / 640

    for x1, y1, x2, y2, conf, cat in res:
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).astype(int)
        x1 = int(x1 * rate)
        y1 = int(y1 * rate)
        x2 = int(x2 * rate)
        y2 = int(y2 * rate)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

if __name__ == '__main__':
    load_model('yolo26n.onnx')
    image = cv2.imread('bdc.jpg')
    result = inference(image)
    draw_boxes(image, result)
    cv2.imshow('title', image)
    cv2.waitKey(0)
