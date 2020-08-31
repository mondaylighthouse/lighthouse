import React, { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const Data = [
  {
    day: 1,
    progress: 0.035,
  },
  {
    day: 2,
    progress: 0.045,
  },
  {
    day: 3,
    progress: 0.088,
  },
  {
    day: 4,
    progress: 0.12,
  },
  {
    day: 5,
    progress: 0.13,
  },
  {
    day: 6,
    progress: 0.16,
  },
  {
    day: 7,
    progress: 0.18,
  },
  {
    day: 8,
    progress: 0.21,
  },
  {
    day: 9,
    progress: 0.22,
  },
  {
    day: 10,
    progress: 0.24,
  },
  {
    day: 11,
    progress: 0.25,
  },
  {
    day: 12,
    progress: 0.27,
  },
  {
    day: 13,
    progress: 0.28,
  },
  {
    day: 14,
    progress: 0.29,
  },
  {
    day: 15,
    progress: 0.31,
  },
  {
    day: 16,
    progress: 0.32,
  },
  {
    day: 17,
    progress: 0.34,
  },
  {
    day: 18,
    progress: 0.35,
  },
  {
    day: 19,
    progress: 0.35,
  },
  {
    day: 20,
    progress: 0.37,
  },
  {
    day: 21,
    progress: 0.39,
  },
  {
    day: 22,
    progress: 0.42,
  },
  {
    day: 23,
    progress: 0.46,
  },
  {
    day: 24,
    progress: 0.47,
  },
  {
    day: 25,
    progress: 0.49,
  },
  {
    day: 26,
    progress: 0.51,
  },
  {
    day: 27,
    progress: 0.52,
  },
  {
    day: 28,
    progress: 0.56,
  },
  {
    day: 29,
    progress: 0.59,
  },
  {
    day: 30,
    progress: 0.6,
  },
  {
    day: 31,
    progress: 0.62,
  },
  {
    day: 32,
    progress: 0.63,
  },
  {
    day: 33,
    progress: 0.66,
  },
  {
    day: 34,
    progress: 0.69,
  },
  {
    day: 35,
    progress: 0.7,
  },
  {
    day: 36,
    progress: 0.71,
  },
  {
    day: 37,
    progress: 0.72,
  },
  {
    day: 38,
    progress: 0.72,
  },
  {
    day: 39,
    progress: 0.74,
  },
  {
    day: 40,
    progress: 0.75,
  },
  {
    day: 41,
    progress: 0.76,
  },
  {
    day: 42,
    progress: 0.77,
  },
  {
    day: 43,
    progress: 0.78,
  },
  {
    day: 44,
    progress: 0.8,
  },
  {
    day: 45,
    progress: 0.81,
  },
  {
    day: 46,
    progress: 0.82,
  },
  {
    day: 47,
    progress: 0.84,
  },
  {
    day: 48,
    progress: 0.85,
  },
  {
    day: 49,
    progress: 0.85,
  },
  {
    day: 50,
    progress: 0.86,
  },
  {
    day: 51,
    progress: 0.82,
  },
  {
    day: 52,
    progress: 0.83,
  },
  {
    day: 53,
    progress: 0.84,
  },
  {
    day: 54,
    progress: 0.85,
  },
  {
    day: 55,
    progress: 0.86,
  },
  {
    day: 56,
    progress: 0.87,
  },
  {
    day: 57,
    progress: 0.88,
  },
  {
    day: 58,
    progress: 0.89,
  },
  {
    day: 59,
    progress: 0.9,
  },
  {
    day: 60,
    progress: 0.92,
  },
  {
    day: 61,
    progress: 0.94,
  },
  {
    day: 62,
    progress: 0.95,
  },
  {
    day: 63,
    progress: 0.96,
  },
  {
    day: 64,
    progress: 0.97,
  },
  {
    day: 65,
    progress: 0.95,
  },
  {
    day: 66,
    progress: 0.93,
  },
  {
    day: 67,
    progress: 0.9,
  },
  {
    day: 68,
    progress: 0.87,
  },
  {
    day: 69,
    progress: 0.85,
  },
  {
    day: 70,
    progress: 0.85,
  },
  {
    day: 71,
    progress: 0.86,
  },
  {
    day: 72,
    progress: 0.88,
  },
  {
    day: 73,
    progress: 0.9,
  },
  {
    day: 74,
    progress: 0.91,
  },
  {
    day: 75,
    progress: 0.93,
  },
  {
    day: 76,
    progress: 0.95,
  },
  {
    day: 77,
    progress: 0.96,
  },
  {
    day: 78,
    progress: 0.99,
  },
  {
    day: 80,
    progress: 0.98,
  },
  {
    day: 81,
    progress: 0.95,
  },
  {
    day: 82,
    progress: 0.93,
  },
  {
    day: 83,
    progress: 0.9,
  },
  {
    day: 84,
    progress: 0.85,
  },
  {
    day: 85,
    progress: 0.81,
  },
  {
    day: 86,
    progress: 0.78,
  },
  {
    day: 87,
    progress: 0.73,
  },
  {
    day: 88,
    progress: 0.7,
  },
  {
    day: 89,
    progress: 0.68,
  },
  {
    day: 90,
    progress: 0.65,
  },
  {
    day: 91,
    progress: 0.62,
  },
  {
    day: 92,
    progress: 0.57,
  },
  {
    day: 93,
    progress: 0.54,
  },
  {
    day: 94,
    progress: 0.49,
  },
  {
    day: 95,
    progress: 0.43,
  },
  {
    day: 96,
    progress: 0.42,
  },
  {
    day: 97,
    progress: 0.41,
  },
];
//const Data = require("./ex2.json");

function Predictionboard() {
  console.log("Hello TensorFlow");

  useEffect(() => {
    // Create an scoped async function in the hook
    async function anyNameFunction() {
      await run();
      //await trainModel();
    }
    // Execute the created function directly
    anyNameFunction();
  }, []);

  let model;
  let cleaned;
  let data;

  async function run() {
    console.log("Hello TensorFlow run");

    cleaned = await Data.map((datensatz) => ({
      vertikal: datensatz.progress, //progress, Miles_per_Gallon
      horizontale: datensatz.day, //day, Horsepower
    })).filter((car) => car.vertikal != null && car.horizontale != null);

    const values = await cleaned.map((d) => ({
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

    data = cleaned;
    console.log("before tensordata");
tf.tidy();
    console.log("Hello TensorFlow getdata" + tensorData);

    model = new tf.sequential({
      layers: [
        await tf.layers.dense({
          units: 450,
          useBias: true,
          inputShape: [1],
        }),
        await tf.layers.dense({
          units: 380,
          activation: "relu",
        }),
        await tf.layers.dense({
          units: 150,
          activation: "sigmoid",
        }),
        await tf.layers.dense({
          units: 1,
          activation: "relu",
          useBias: true,
        }),
      ],
    });
    console.log("Hello TensorFlow createmodel" + model);
    await tfvis.show.modelSummary({ name: "Model Summary" }, model);
    console.log("Hello TensorFlow convert");
    const { inputs, labels } = tensorData;
    console.log("check" + model);
    trainModel(model, inputs, labels);
    console.log("Done Training");
    testModel(model, data, tensorData);
  }

  const tensorData = await tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);
    console.log("shuffle tensordata" + data);
    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horizontale);
    const labels = data.map((d) => d.vertikal);
    console.log("step 2 finifshed" + inputs);
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    console.log("step 3 finifshed" + inputTensor);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    console.log("step 4 finifshed" + labelTensor);
    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();
    console.log("step 5 finifshed" + labelTensor);
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




  async function trainModel(model, inputs, labels) {
    const batchSize = 35; //32
    const epochs = 50; //80
    const learningRate = 0.002;
    const momentum = 0.5;
    console.log("Hello TensorFlow trainmodel");

    await model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: tf.losses.meanSquaredError,
      metrics: ["mae"],
    });

    return model.fit(inputs, labels, {
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

  function testModel(model, inputData, normalizationData) {
    console.log("Hello TensorFlow testmodel");
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
    const [xs, preds] = tf.tidy(() => {
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));
      const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
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
    console.log("TensorFlow testmodel" + model);
  }

  return (
    <div className="App">
      <h1>PREDICTION BOARD</h1>
    </div>
  );
}

export default Predictionboard;
