# quantize.py
import os
import argparse
import onnx
from onnxconverter_common import float16

def quantize_and_save(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    quantized_model_path = os.path.join(output_dir, "resnet50_fp16.onnx")

    # Load original ONNX model
    model = onnx.load(os.path.join(input_dir, "resnet50.onnx"))
    # Convert all weights to FP16
    model_fp16 = float16.convert_float_to_float16(model)

    # Save FP16 model
    onnx.save(model_fp16, quantized_model_path)
    print(f"✅ FP16 model saved to: {quantized_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    quantize_and_save(args.input_dir ,args.output_dir)