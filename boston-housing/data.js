const fs = require('fs')
const Papa = require('papaparse')
const tf = require('@tensorflow/tfjs-node-gpu')

/**
 * 数据处理
 */

const file = fs.readFileSync('./HousingData.csv', 'utf8')
const results = Papa.parse(file, { header: true })

const data = results.data.map(row => Object.keys(row)
    .map(key => parseFloat(row[key])))
    .filter(row => row.every(value => !isNaN(value)))

const trainFeatures = data.map(row => row.slice(0, 13))
const trainTarget = data.map(row => row[13])

const rawTrainFeaturesTensor = tf.tensor2d(trainFeatures)
const trainTargetTensor = tf.expandDims(tf.tensor1d(trainTarget), 1)

// 房价平均值
const avgPrice = tf.mean(trainTargetTensor)
console.log('Average price:', avgPrice.dataSync()[0]) // 22.35964584350586
const baseline = tf.mean(tf.pow(tf.sub(trainTargetTensor, avgPrice), 2))
console.log('Baseline loss:', baseline.dataSync()[0]) // 83.38188934326172


// 标准化
const meanTrainFeatures = tf.mean(rawTrainFeaturesTensor, 0).dataSync()
console.log('平均值：', meanTrainFeatures)
const stdTrainFeatures = tf.sqrt(tf.mean(tf.pow(tf.sub(trainFeatures, meanTrainFeatures), 2), 0))
console.log('标准差：', meanTrainFeatures)

const trainFeaturesTensor = rawTrainFeaturesTensor.sub(meanTrainFeatures).div(stdTrainFeatures)

module.exports = { trainFeaturesTensor, trainTargetTensor }