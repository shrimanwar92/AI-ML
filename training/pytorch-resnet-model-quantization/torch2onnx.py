import torch
import torchvision.models as models
import os
import argparse


def convert(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # Dummy input for export (batch size 1, 3 channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_model_path = f"{output_dir}/resnet50.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )

    print(f"✅ Exported resnet50 to {onnx_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    convert(args.output_dir)