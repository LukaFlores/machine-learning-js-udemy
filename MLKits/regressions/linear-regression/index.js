require("@tensorflow/tfjs-node-gpu");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv.js");
const LinearRegression = require("./linear-regression");

const path = require("path");
const filePath = path.join(__dirname, "../data/cars.csv");

let { features, labels, testFeatures, testLabels } = loadCSV(filePath, {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.01,
  iterations: 100,
});

regression.train();

console.log("M:", regression.m, "B:", regression.b);
