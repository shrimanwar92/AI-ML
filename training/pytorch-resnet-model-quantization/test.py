# test_onnx_models.py
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from torchvision import transforms
import argparse
from io import BytesIO

def test(input_dir):
    ORIGINAL_MODEL_PATH = os.path.join(input_dir, "resnet50.onnx")
    FP16_MODEL_PATH = os.path.join(input_dir, "resnet50_fp16.onnx")

    image_urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png",
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://cdn.pixabay.com/photo/2023/04/21/15/42/portrait-7942151_1280.jpg",
        "https://static.vecteezy.com/system/resources/previews/010/938/844/non_2x/tropical-purple-butterfly-illustration-beautiful-butterfly-vector.jpg",
        "https://media.istockphoto.com/id/479842074/photo/empty-road-at-building-exterior.jpg?s=612x612&w=0&k=20&c=SbyfZGN0i2O_QPLCdBcu9vhuzbQvTz4bGEn-lIzrN0E="
    ]

    # Preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image and convert to RGB, return as float32 or float16
    def load_image(url, dtype=np.float32):
        headers = {"User-Agent": "ml-testing-bot/1.0 (email@example.com)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # raise if download failed
        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).numpy()
        return tensor.astype(dtype)

    # Load ONNX models
    original_sess = ort.InferenceSession(ORIGINAL_MODEL_PATH, providers=["CPUExecutionProvider"])
    fp16_sess = ort.InferenceSession(FP16_MODEL_PATH, providers=["CPUExecutionProvider"])

    # Load ImageNet labels
    imagenet_labels = requests.get(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    ).text.split("\n")

    # Test loop
    for url in image_urls:
        # Original float32 model
        input_tensor32 = load_image(url, dtype=np.float32)
        orig_out = original_sess.run(None, {original_sess.get_inputs()[0].name: input_tensor32})
        orig_class = np.argmax(orig_out[0])

        # FP16 model
        input_tensor16 = load_image(url, dtype=np.float16)
        fp16_out = fp16_sess.run(None, {fp16_sess.get_inputs()[0].name: input_tensor16})
        fp16_class = np.argmax(fp16_out[0])

        # Compare predictions
        print(f"\nImage: {url}")
        print(f"Original ONNX (float32): {imagenet_labels[orig_class]} ({orig_class})")
        print(f"FP16 ONNX: {imagenet_labels[fp16_class]} ({fp16_class})")
        print(f"Prediction match? {orig_class == fp16_class}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    args = parser.parse_args()
    test(args.input_dir)
