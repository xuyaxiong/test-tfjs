const tf = require('@tensorflow/tfjs-node-gpu')
const Papa = require('papaparse')
const fs = require('fs')

/**
 * 处理鸢尾花数据集
 */

const csvStr = fs.readFileSync('./iris.csv', encoding = 'utf8')
const result = Papa.parse(csvStr, { header: true })

// 类别 setosa versicolor virginica
const data = result.data
    .map(row => Object.keys(row).map(key => row[key]))
    .map(row => {
        let newRow = row.slice(1, 5).map(cell => parseFloat(cell))
        let target
        switch (row[5]) {
            case 'setosa':
                target = 0
                break
            case 'versicolor':
                target = 1
                break
            case 'virginica':
                target = 2
                break
        }
        newRow.push(target)
        return newRow
    })

const rawTrainFeatures = data.map(row => row.slice(0, 4))
const trainTarget = data.map(row => row[4])

const trainFeaturesTensor = tf.tensor2d(rawTrainFeatures)
const trainTargetTensor = tf.oneHot(tf.tensor1d(trainTarget, 'int32'), 3)

module.exports = {
    trainFeaturesTensor,
    trainTargetTensor
}