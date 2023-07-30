const tf = require('@tensorflow/tfjs-node-gpu')

const xs = [1, 2, 3, 4, 5]
const ys = xs.map(x => {
    let bias = Math.random() - 0.5
    return 20 * x + 100 + bias
})

const xs_tensor = tf.tensor2d(xs, [xs.length, 1])
const ys_tensor = tf.tensor2d(ys, [ys.length, 1])

// 学习率
const LEARNING_RATE = 0.01

// 模型
const model = tf.sequential()
model.add(tf.layers.dense({ inputShape: [1], units: 1 }))

// 优化器
const optimizer = tf.train.sgd(LEARNING_RATE)

model.compile({
    loss: 'meanSquaredError',
    optimizer: optimizer
})

model.summary()

model.fit(xs_tensor, ys_tensor, {
    epochs: 2000,
    callbacks: {
        onBatchEnd: async (epoch, logs) => {
            console.log('k =', model.layers[0].getWeights()[0].dataSync()[0])
            console.log('b =', model.layers[0].getWeights()[1].dataSync()[0])
        }
    }
})