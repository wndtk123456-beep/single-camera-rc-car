from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


sess = None


def load_model(model_path):
    global sess
    sess = ort.InferenceSession(model_path)


def inference(image):
    if sess is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    image = image[..., [2, 1, 0]]  # BGR to RGB
    back = np.zeros((640, 640, 3), np.uint8)

    height, width, _ = image.shape
    if height > width:
        resized_height = 640
        resized_width = width * 640 // height
    else:
        resized_width = 640
        resized_height = height * 640 // width

    resized = cv2.resize(image, (resized_width, resized_height))
    back[0:resized_height, 0:resized_width] = resized

    model_input = back.astype(np.float32) / 255
    model_input = model_input.transpose(2, 0, 1)[None]

    result = sess.run(None, input_feed={"images": model_input})[0][0]
    return result


def draw_boxes(image, result):
    height, width, _ = image.shape
    rate = max(height, width) / 640

    for x1, y1, x2, y2, conf, cat in result:
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).astype(int)
        x1 = int(x1 * rate)
        y1 = int(y1 * rate)
        x2 = int(x2 * rate)
        y2 = int(y2 * rate)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parents[2] / "models" / "last.onnx"
    sample_path = Path("sample.jpg")

    load_model(str(model_path))
    image = cv2.imread(str(sample_path))
    if image is None:
        raise FileNotFoundError("Place sample.jpg in the current directory before running this test.")

    result = inference(image)
    draw_boxes(image, result)
    cv2.imshow("result", image)
    cv2.waitKey(0)
