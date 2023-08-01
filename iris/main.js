const model = require('./model')
const { trainFeaturesTensor, trainTargetTensor } = require('./data')

/**
 * 鸢尾花分类
 */

model.fit(trainFeaturesTensor, trainTargetTensor, {
    epochs: 100,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
        }
    }
})