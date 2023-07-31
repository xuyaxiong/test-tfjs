/**
 * 波士顿房价预测
 */

const trainData = require('./data')
const model = require('./model')
const tf = require('@tensorflow/tfjs-node-gpu')

const { trainFeaturesTensor, trainTargetTensor } = trainData
const NUM_EPOCHS = 200
const BATCH_SIZE = 40
model.fit(trainFeaturesTensor, trainTargetTensor, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            const weights = tf.squeeze(model.layers[0].getWeights()[0])
            const { values, indices } = tf.topk(weights, 13, true)
            values.print() // 权重
            indices.print() // 下标
        }
    }
})