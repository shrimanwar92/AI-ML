## Model Optimization Techniques

### Reducing Model Size
There are two ways to reduce the size of your model:
- Model Quantization
- Model Pruning

#### Model Quantization
- The most popular way to reduce model size is a process called quantization. 
- Models hold millions, or even billions, or weights and biases serving as the model’s memory. These parameters, typically stored as 32-bit floating-point numbers, result in models that require massive amounts of memory.
- Quantization works by reducing the precision of the weights — essentially simplifying these parameters into more compact forms and truncating the length of the digits. This process drastically reduces the memory footprint and computational requirements of the model.
- There is a drawback to quantization: a reduction in model accuracy. 

#### Model Pruning
- Model pruning is the technique of removing neurons within the neural network that do not improve a model’s performance.
- Pruning these unimportant parameters reduces the number of weights which, effectively, makes the model smaller.
- In addition to decreasing the model size, this can reduce the inference latency as the model has fewer calculations to execute. The drawback is the potential for loss in accuracy.

