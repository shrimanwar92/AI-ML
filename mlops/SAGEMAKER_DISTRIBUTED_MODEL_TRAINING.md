## Sagemaker distributed training

### Data parallelism = split the data, keep copies of the model.
### Model parallelism = split the model, send data through parts of it.

---

### 🔹 1. Data Parallelism with tf.distribute.MirroredStrategy
```python
import tensorflow as tf

# Use MirroredStrategy for data parallel training on multiple GPUs
strategy = tf.distribute.MirroredStrategy()

print("Number of devices:", strategy.num_replicas_in_sync)

# Create model inside strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

# Dummy dataset (MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Train (data automatically split across GPUs)
model.fit(x_train, y_train, epochs=5, batch_size=256)
```
- 👉 Here, TensorFlow automatically splits the batch across GPUs, and then synchronizes weight updates after each step.

---

### 🔹 2. Model Parallelism (Manual Partitioning)
TensorFlow doesn’t have automatic model parallelism like data parallelism. Instead, you manually place layers on different devices (GPUs/TPUs).
```python
import tensorflow as tf

# Example: Split model across two GPUs
with tf.device("/GPU:0"):
    input_layer = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(512, activation="relu")(input_layer)

with tf.device("/GPU:1"):
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Dummy dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0

# Train (data flows GPU0 -> GPU1)
model.fit(x_train, y_train, epochs=5, batch_size=256)
```

```python
import os
import argparse
import tensorflow as tf
import smdistributed.modelparallel.tensorflow as smp

# Initialize SMP
smp.init()

def build_partitioned_model():
    inputs = tf.keras.Input(shape=(784,))
    with smp.partition(0):
        x = tf.keras.layers.Dense(4096, activation="relu")(inputs)
        x = tf.keras.layers.Dense(4096, activation="relu")(x)
    with smp.partition(1):
        x = tf.keras.layers.Dense(2048, activation="relu")(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

def main(args):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test  = x_test.reshape(-1, 784) / 255.0

    model = build_partitioned_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    model.evaluate(x_test, y_test, verbose=2)

    model.save(os.path.join(os.environ["SM_MODEL_DIR"], "1"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()
    main(args)
```
`Notes:`
- You manually assign layers to partitions/GPU groups via smp.partition(i).
- Works best on multi-GPU instances (e.g., ml.p3.16xlarge, ml.p4d.24xlarge).
- 👉 Here, the first layer lives on GPU:0/smp.partition(0) and the later layers on GPU:1/smp.partition(1). The data flows through them sequentially. This helps when the model is too large for one GPU.

---

### 🔹 3. Sagemakers built in distributed training example
ScriptProcessor —  won’t give you SageMaker’s built-in distributed training support. We will use an Estimator + TrainingStep, so you can use distribution configs and scale out training. Below are two helpers you can drop into your pipeline.py to create steps for data parallel and model parallel training. They assume you have ROLE, local_session, and your S3 bucket constants already defined (as in your current file).
```python
from sagemaker.workflow.steps import TrainingStep
from sagemaker.tensorflow import TensorFlow

# --- Data-parallel (single-node, multi-GPU via MirroredStrategy) ---
def train_tf_data_parallel_step(output_s3_prefix: str):
    estimator = TensorFlow(
        entry_point="train_tf_dp.py",
        source_dir="scripts",
        role=ROLE,
        instance_type="ml.p3.8xlarge",    # 4x V100 GPUs (example)
        instance_count=1,                 # single node; MirroredStrategy uses all GPUs
        framework_version="2.12",
        py_version="py39",
        hyperparameters={"epochs": 5, "batch-size": 256},
        output_path=f"s3://{S3_BUCKET}/{output_s3_prefix}",
        sagemaker_session=local_session,
    )

    return TrainingStep(
        name="TrainTFDataParallel",
        estimator=estimator,
        inputs={}   # not using channels here; the script downloads MNIST
    )

# --- Data-parallel (multi-node with Horovod) ---
def train_tf_data_parallel_hvd_step(output_s3_prefix: str):
    estimator = TensorFlow(
        entry_point="train_tf_dp_hvd.py",
        source_dir="scripts",
        role=ROLE,
        instance_type="ml.p3.8xlarge",
        instance_count=2,                 # multi-node
        framework_version="2.12",
        py_version="py39",
        hyperparameters={"epochs": 5, "batch-size": 256, "lr": 0.001},
        distribution={
            "mpi": {
                "enabled": True,
                "processes_per_host": 4,   # GPUs per node
                "custom_mpi_options": "-x NCCL_DEBUG=INFO"
            }
        },
        output_path=f"s3://{S3_BUCKET}/{output_s3_prefix}",
        sagemaker_session=local_session,
    )

    return TrainingStep(
        name="TrainTFDataParallelHvd",
        estimator=estimator,
        inputs={}
    )

# --- Model-parallel (SMP) ---
def train_tf_model_parallel_step(output_s3_prefix: str):
    estimator = TensorFlow(
        entry_point="train_tf_mp.py",
        source_dir="scripts",
        role=ROLE,
        instance_type="ml.p3.16xlarge",   # multi-GPU required
        instance_count=1,
        framework_version="2.12",
        py_version="py39",
        distribution={
            "smdistributed": {
                "modelparallel": {
                    "enabled": True,
                    "parameters": {
                        "partitions": 2,           # number of partitions (GPUs)
                        "pipeline": "interleaved", # pipeline schedule
                        "microbatches": 4,         # pipeline micro-batching
                        "active_microbatches": 2,
                        "ddp": False               # DDP+SMP = hybrid (set True if you want both)
                    }
                }
            }
        },
        hyperparameters={"epochs": 3, "batch-size": 256},
        output_path=f"s3://{S3_BUCKET}/{output_s3_prefix}",
        sagemaker_session=local_session,
    )

    return TrainingStep(
        name="TrainTFModelParallel",
        estimator=estimator,
        inputs={}
    )
```
