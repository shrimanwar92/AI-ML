// create a model
// diagram of neural network is my empty model
const model = tf.sequential();

// create a hidden layer
// dense is a "fully connected" layer
const hidden = tf.layers.dense({
    units: 4,
    activation: "sigmoid",
    inputShape: [2]
});
// add layer to model
model.add(hidden);

// create another layer
const output = tf.layers.dense({
    units: 1,
    // here the input share is "inferred from previous layer"
    activation: "sigmoid"
});
model.add(output);

// an optimizer using gradient descent
// i'am done configuring the model so compiling it
model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1]
])

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
])

async function train() {
    const config = {
        shuffle: true,
        epochs: 10
    }
    for (let i=0; i<1000; i++) {
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0]);
    }
}

train().then(() => {
    console.log("training complete");
    let outputs = model.predict(xs);
    outputs.print();
})