const tf = require('@tensorflow/tfjs-node-gpu')

const model = tf.sequential()
model.add(tf.layers.dense({
    inputShape: [4],
    units: 10,
    activation: 'sigmoid'
}))
model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
}))
model.summary()

const LEARNING_RATE = 0.01
const optimizer = tf.train.adam(LEARNING_RATE)
model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
})


module.exports = model