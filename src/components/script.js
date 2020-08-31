console.log("Hello TensorFlow");
const faktor = 100; //ion
const batchSize = 45; //32
const epochs = 40; //80

async function run() {
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);
  const data = await getData();
  const datatoanalyze = await getData2();
  const values = data.map((d) => ({
    x: d.horizontale,
    y: d.vertikal,
  }));

  tfvis.render.scatterplot(
    { name: "Prediction project progress" },
    { values },
    {
      xLabel: "DAYS",
      yLabel: "Progress in %",
      height: 300,
    }
  );

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const tensorData2 = convertToTensor(datatoanalyze);
  const { inputs, labels } = tensorData;
  const { inputs2, labels2 } = tensorData2;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");
  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, datatoanalyze, tensorData2);
}

async function getData() {
  const DataReq = await fetch("ex2.json");
  const Data = await DataReq.json();
  const cleaned = Data.map((datensatz) => ({
    vertikal: datensatz.progress, //progress, Miles_per_Gallon
    horizontale: datensatz.day, //day, Horsepower
  })).filter(
    (datensatz) => datensatz.vertikal != null && datensatz.horizontale != null
  );

  return cleaned;
}

//here call the second data to analyze
async function getData2() {
  const DataReq = await fetch("ex2.json");
  const Data = await DataReq.json();
  const cleaned = Data.map((datensatz) => ({
    vertikal: datensatz.progress, //progress, Miles_per_Gallon
    horizontale: datensatz.day, //day, Horsepower
  })).filter(
    (datensatz) => datensatz.vertikal != null && datensatz.horizontale != null
  );

  return cleaned;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
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
  const learningRate = 0.002;
  const momentum = 0.5;

  const decay = 0.01;
  const beta_1 = 0.9;
  const beta_2 = 1.1;
  const epsilon = 1e-7;
  const amsgrad = "false";
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adamax(learningRate),
    //.adam (learningRate?, beta1?, beta2?, epsilon?)
    //.sgd(learningRate),
    //.momentum (learningRate, momentum, useNesterov?),
    //.adagrad (learningRate, initialAccumulatorValue?),
    //.adadelta (learningRate?, rho?, epsilon?),
    //.adamax (learningRate?, beta1?, beta2?, epsilon?, decay?) ,
    //.rmsprop (learningRate, decay?, momentum?, epsilon?, centered?)
    loss: tf.losses.meanSquaredError,
    //
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
      { name: "Training Performance" },
      ["loss", "mae"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();
  console.log("halt");
  // Add a single input layer
  // ok model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  // ok model.add(tf.layers.dense({ units: 60, })); //24

  // Add an output layer
  // ok model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', useBias: true }));
  const inputLayer = tf.layers.dense({
    kernel_initializer: "glorot_uniform",
    units: 450,
    useBias: true,
    inputShape: [1],
    activation: "linear",
    dtype: "float32",
  });
  const hiddenLayer = tf.layers.dense({ units: 380, activation: "relu" });
  const hiddenLayer2 = tf.layers.dense({ units: 150, activation: "softsign" });
  const outputLayer = tf.layers.dense({
    units: 1,
    activation: "softsign",
    useBias: true,
  }); //"sigmoid", 'softmax', "relu", 'softplus', 'linear' , useBias: true
  model.add(inputLayer);
  model.add(hiddenLayer);
  model.add(hiddenLayer2);
  model.add(outputLayer);
  return model;
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 120);
    const preds = model.predict(xs.reshape([120, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    console.log(preds[i] + " " + i + " " + val);
    return { x: val, y: preds[i] }; //jeff
  });
  console.log(predictedPoints.length);
  const originalPoints = inputData.map((d) => ({
    x: d.horizontale,
    y: d.vertikal,
  }));

  tfvis.render.linechart(
    //scatterplot
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
    { name: "Model Predictions" },
    { values: [predictedPoints], series: ["predicted"] },
    {
      xLabel: "DAYS",
      yLabel: "Values in %",
      height: 300,
    }
  );
}

document.addEventListener("DOMContentLoaded", run);
