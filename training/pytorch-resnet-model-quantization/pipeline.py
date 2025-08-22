import os
from sagemaker.processing import ProcessingInput, ProcessingOutput
from helpers import (DEFAULT_REGION, S3_BUCKET, 
                    get_pytorch_processor,
                    local_session, ROLE)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

# Dummy AWS credentials
os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION

def pytorch_to_onnx():
    torch_processor = get_pytorch_processor("convert_pytorch_model_to_onnx")
    outputs = ("/opt/ml/processing/output/model", f"s3://{S3_BUCKET}/quantized")
   
    return ProcessingStep(
        name="Pytorch2Onnx",
        processor=torch_processor,
        outputs=[
            ProcessingOutput(source=outputs[0], destination=outputs[1], output_name="onnx_model"),  
        ],
        code="torch2onnx.py",
        job_arguments=["--output-dir", outputs[0]],
    )

def quantize(onnx_step: ProcessingStep):
    torch_processor = get_pytorch_processor("quantize")
    inputs = ("/opt/ml/processing/input", onnx_step.properties.ProcessingOutputConfig.Outputs["onnx_model"].S3Output.S3Uri)
    outputs = ("/opt/ml/processing/output", f"s3://{S3_BUCKET}/quantized")

    return ProcessingStep(
        name="QuantizeOnnxModel",
        processor=torch_processor,
        code="quantize.py",   # script shown below
        inputs=[ProcessingInput(source=inputs[1], destination=inputs[0])],
        outputs=[ProcessingOutput(source=outputs[0], destination=outputs[1], output_name="quantized_model")],
        job_arguments=["--input-dir", inputs[0], "--output-dir", outputs[0]],
    )

def test_quantized_model():
    torch_processor = get_pytorch_processor("test_model")
    inputs = ("/opt/ml/processing/input", f"s3://{S3_BUCKET}/quantized")
   
    return ProcessingStep(
        name="TestQuantizedModel",
        processor=torch_processor,
        code="test.py",
        inputs=[
            ProcessingInput(source=inputs[1], destination=inputs[0])  
        ],
        job_arguments=["--input-dir", inputs[0]],
    )


onnx_step = pytorch_to_onnx()
quant_step = quantize(onnx_step)
test_step = test_quantized_model()

# Build pipeline
pipeline = Pipeline(
    name="QuantizeModelPipeline",
    steps=[
        onnx_step, 
        quant_step,
        test_step
    ],
    sagemaker_session=local_session
)

pipeline.upsert(role_arn=ROLE)
execution = pipeline.start()