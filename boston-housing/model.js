const tf = require('@tensorflow/tfjs-node-gpu')

// 模型
const model = tf.sequential()
model.add(tf.layers.dense({ inputShape: [13], units: 1 }))
const LEARNING_RATE = 0.01
model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
})
model.summary()

module.exports = model