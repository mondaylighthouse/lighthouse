import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";



async function run(ProzFromMonday, DatesFromMonday) {
  let listDates = DatesFromMonday;
  let listProZent = ProzFromMonday;

  console.log("run" + listDates + listProZent);

  const model = createModel();
  //tfvis.show.modelSummary({ name: "Model Summary" }, model);
  const data = await getData();
  const values = data.map((d) => ({
    x: d.horizontale,
    y: d.vertikal,
  }));

  // tfvis.render.scatterplot(
  //  { name: "Prediction project progress" },
  //  { values },
  //  {
  //     xLabel: "DAYS",
  //     yLabel: "Progress in %",
  //     height: 300,
  //   }
  //  );

  async function getData() {
    // const DataReq = await fetch(tesst);
    // const Data2 = await DataReq.json();
    const cleaned = await ProzFromMonday.map((datensatz, index) => ({
      vertikal: datensatz * 1, //progress, Miles_per_Gallon
      horizontale: index + 1, //day, Horsepower
    })).filter((car) => car.vertikal != null && car.horizontale != null);

    return cleaned;
  }

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();
  const inputLayer = tf.layers.dense({
    units: 450,
    useBias: true,
    activation: "linear",
    inputShape: [1],
  });
  const hiddenLayer = tf.layers.dense({
    units: 380,
    activation: "relu6",
    useBias: true,
  });
  const hiddenLayer2 = tf.layers.dense({
    units: 150,
    activation: "sigmoid",
    useBias: true,
  });
  const outputLayer = tf.layers.dense({
    units: 1,
    activation: "relu6",
    useBias: true,
  }); //"sigmoid", 'softmax', "relu"  , useBias: true
  model.add(inputLayer);
  model.add(hiddenLayer);
  model.add(hiddenLayer2);
  model.add(outputLayer);
  return model;
}

function convertToTensor(data) {
  console.log("Hello TensorFlow convert");
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horizontale);
    const labels = data.map((d) => d.vertikal);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  const batchSize = 35; //32
  const epochs = 60; //80
  const learningRate = 0.003;
  const momentum = 0.5;
  console.log("Hello TensorFlow trainmodel");
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(learningRate),
    //.adam (learningRate?, beta1?, beta2?, epsilon?)
    //.sgd(learningRate),
    //.momentum (learningRate, momentum, useNesterov?),
    //.adagrad (learningRate, initialAccumulatorValue?),
    //.adadelta (learningRate?, rho?, epsilon?),
    //.adamax (learningRate?, beta1?, beta2?, epsilon?, decay?) ,
    //.rmsprop (learningRate, decay?, momentum?, epsilon?, centered?)
    loss: tf.losses.meanSquaredError,
    //losses = tfp.math.minimize(loss_fn, num_steps=100, optimizer=tf.optimizers.Adam(learning_rate=0.1))
    //.cosineDistance (labels, predictions, axis, weights?, reduction?)
    //.absoluteDifference (labels, predictions, weights?, reduction?)
    //.hingeLoss (labels, predictions, weights?, reduction?)
    //.huberLoss (labels, predictions, weights?, delta?, reduction?)
    //.logLoss (labels, predictions, weights?, epsilon?, reduction?)
    //.meanSquaredError (labels, predictions, weights?, reduction?)
    //.sigmoidCrossEntropy (multiClassLabels, logits, weights?, labelSmoothing?, reduction?)
    //.softmaxCrossEntropy (onehotLabels, logits, weights?, labelSmoothing?, reduction?)
    //.computeWeightedLoss (losses, weights?, reduction?)
    metrics: ["mae"], //metrics: , ["accuracy"], ['mse'] ['mae']
  });

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance /loss" },
      ["loss"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 150);
    const preds = model.predict(xs.reshape([150, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }; //jeff
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horizontale,
    y: d.vertikal,
  }));

  tfvis.render.scatterplot(
    //scatterplot //linechart
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "DAYS",
      yLabel: "Values in %",
      height: 300,
    }
  );
  tfvis.render.linechart(
    //scatterplot //linechart
    { name: "Model Predictions" },
    {
      values: [predictedPoints],
      series: ["predicted"],
    },
    {
      xLabel: "DAYS",
      yLabel: "Values in %",
      height: 300,
    }
  );
}

function Predictionboard() {
  let query = "{boards(limit:1) { items { column_values{text } } } }";
  //let query = "{ TT1 { Date Progress } }";
  const [dataFromMonday, setdataFromMonday] = useState(0);

  useEffect(() => {
    fetch("https://api.monday.com/v2", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
        Authorization:
          "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjczNzc5NzY0LCJ1aWQiOjE0OTg2NjQ1LCJpYWQiOiIyMDIwLTA4LTI4VDIyOjE2OjUxLjAwMFoiLCJwZXIiOiJtZTp3cml0ZSJ9.QFmtxR0qc1ImZlKoexckicw8hAl8cFJk_MaltdNjhOs",
      },
      body: JSON.stringify({
        query: query,
      }),
    })
      .then((res) => res.json())
      .then((res) => setdataFromMonday(res.data.boards[0].items));
  }, [query]);

  const oArr = Object.entries(dataFromMonday);
  const DatesFromMonday = [];
  const ProzFromMonday = [];
  oArr.forEach(([key, value]) => {
    DatesFromMonday.push(value.column_values[0].text); // 'one'
    ProzFromMonday.push(value.column_values[1].text); // 'one'
  });
  //console.log("where proz" + ProzFromMonday);
  //console.log("where dates" + DatesFromMonday);
  run(ProzFromMonday, DatesFromMonday);
  return (
    <div className="App">
      <h1>PREDICTION BOARD</h1>
      <div id="here"></div>
    </div>
  );
}

export default Predictionboard;
