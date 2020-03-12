using NN_Samples.Common;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class PerceptronTrainer
    {
        public TrainStats Train(IPerceptron perceptron, TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError)
        {
            double error = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = maxEpoch >= 100 ? maxEpoch / 100 : maxEpoch > 10 ? maxEpoch / 10 : 1;
            double[,] outputs = new double[rowCountY, columnCountY];
            int epoch = 0;
            do
            {
                epoch++;
                for (int s = 0; s < trainData.Inputs.GetLength(0); s++)
                {
                    double[] input = new double[columnCountX];
                    double[] targetOutputs = new double[columnCountY];
                    for (var c = 0; c < columnCountX; c++)
                    {
                        input[c] = trainData.Inputs[s, c];
                    }
                    for (var c = 0; c < columnCountY; c++)
                    {
                        targetOutputs[c] = trainData.Outputs[s, c];
                    }
                    var currentOutput = perceptron.FeedForward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs[s, c] = currentOutput[c];
                    }
                    perceptron.BackPropagation(input, targetOutputs, currentOutput, learningRate);
                }
                error = CommonFunctions.MeanBatchMSE(outputs, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("({0}) Mean Batch MSE: {1}", epoch, error);
                }
            }
            while (error > targetError && epoch < maxEpoch);
            TrainStats trainStats = new TrainStats
            {
                LastError = error,
                NumberOfEpoch = epoch
            };
            return trainStats;
        }

        public TrainStats[] PairTrain(IPerceptron perceptron1, IPerceptron perceptron2, TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError)
        {
            double error1 = double.MaxValue;
            double error2 = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = maxEpoch >= 100 ? maxEpoch / 100 : maxEpoch > 10 ? maxEpoch / 10 : 1;
            double[,] outputs1 = new double[rowCountY, columnCountY];
            double[,] outputs2 = new double[rowCountY, columnCountY];
            int epoch = 0;
            do
            {
                epoch++;
                for (int s = 0; s < trainData.Inputs.GetLength(0); s++)
                {
                    double[] input = new double[columnCountX];
                    double[] targetOutputs = new double[columnCountY];
                    for (var c = 0; c < columnCountX; c++)
                    {
                        input[c] = trainData.Inputs[s, c];
                    }
                    for (var c = 0; c < columnCountY; c++)
                    {
                        targetOutputs[c] = trainData.Outputs[s, c];
                    }
                    var currentOutput1 = perceptron1.FeedForward(input);
                    var currentOutput2 = perceptron2.FeedForward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs1[s, c] = currentOutput1[c];
                        outputs2[s, c] = currentOutput1[c];
                    }
                    perceptron1.BackPropagation(input, targetOutputs, currentOutput1, learningRate);
                    perceptron2.BackPropagation(input, targetOutputs, currentOutput2, learningRate);
                }
                error1 = CommonFunctions.GeneralError(outputs1, trainData.Outputs);
                error2 = CommonFunctions.GeneralError(outputs2, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("P1({0}) Error: {1}", epoch, error1);
                    Console.WriteLine("P2({0}) Error: {1}", epoch, error2);
                }
            }
            while (error1 > targetError && error2 > targetError && epoch < maxEpoch);
            TrainStats[] trainStats = new TrainStats[2];
            trainStats[0].LastError = error1;
            trainStats[0].NumberOfEpoch = epoch;
            trainStats[0].LastError = error2;
            trainStats[0].NumberOfEpoch = epoch;
            return trainStats;
        }
    }
}
