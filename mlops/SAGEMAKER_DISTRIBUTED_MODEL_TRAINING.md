## Sagemaker distributed training

### Data parallelism = split the data, keep copies of the model.
### Model parallelism = split the model, send data through parts of it.

---

### Why does data source matter with SageMaker?
- Where and how our data is provided is essential to optimizing training time.
- `File Mode:` Here SageMaker downloads your dataset into the instance memory before training kicks off. Downloads full data in memory.
- `Fast File Mode: ` With Fast File Mode the dataset is streamed into the instance in real-time so we can avoid the overhead of downloading the entire dataset.
- `Fsx Lustre: ` Outside of S3 there’s also options to work with Elastic File System (EFS) and FsX Lustre on SageMaker. FsX Lustre you can scale at a greater rate compared to other options, but there is operational overhead of setting up the VPC for this option.

---

### Instance type selection
- For Computer Vision and NLP use GPU based instances and for algorithm like XGBoost use memory optimized instances such as `ml.m5.24xlarge`.
---

### Training Input (`TrainingInput` class in sagemaker)
```python
train_input = TrainingInput('s3://sagemaker-us-east-1-474422712127/xgboost-1TB/', content_type="text/csv", 
input_mode='FastFile', distribution = "ShardedByS3Key")
training_path
```
- we specify that we are utilizing `FastFile` mode, otherwise it defaults to File mode.
- We also specify the distribution as `ShardedByS3Key`, this indicates we want to distribute all our different S3 files across all instances. Otherwise all data files will get loaded into each and every single instance leading to a much longer training time.

---

### XGBoost estimator
- we specify our instance count to be 25
- Once we specify a count greater than one, SageMaker infers Distributed Data Parallel for our model.
```python

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region='us-east-1',
        version="1.0-1",
        py_version="py3",
        instance_type='ml.m5.24xlarge',
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type='ml.m5.24xlarge',
        instance_count=25,
        output_path=f's3://{default_bucket}/{s3_prefix}/xgb_model',
        sagemaker_session=sagemaker_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )
  ```
  - We can then kick off a training job by fitting the algorithm on the training input.
  ```python
    xgb_train.fit({'train': train_input})
  ```
---

#### Outside of tuning the hardware behind the training job we can revisit the data source format we were talking about. You can evaluate FsX Lustre which can scale to 100s of GB/s throughput. Another option is sharding the dataset in a different format like parquet to try various combinations of number of files and file size.

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

