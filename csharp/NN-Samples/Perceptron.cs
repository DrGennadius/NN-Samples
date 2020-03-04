using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples
{
    public class Perceptron
    {
        Layer[] Layers;

        public Perceptron(params int[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];
            Random r = new Random();

            for (int i = 0; i < neuronsPerLayer.Length; i++)
            {
                Layers[i] = new Layer(neuronsPerLayer[i], i == 0 ? neuronsPerLayer[i] : neuronsPerLayer[i - 1], r);
            }
        }

        public double[] FeedForward(double[] input)
        {
            double[] outputs = new double[0];
            for (int i = 1; i < Layers.Length; i++)
            {
                outputs = Layers[i].FeedForward(input);
                input = outputs;
            }
            return outputs;
        }

        public void BackPropagation(double[] input, double[] goalOutput, double[] realOutput, double learningRate)
        {
            double[][] errors = new double[Layers.GetLength(0)][];
            int layerCount = Layers.GetLength(0);

            // From end.
            errors[layerCount - 1] = new double[Layers[layerCount - 1].Neurons.Length];
            for (int i = 0; i < goalOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                errors[layerCount - 1][i] = goalOutput[i] - realOutput[i];
            }

            // Looping relative to the previous layer to the next layer.
            for (int i = layerCount - 2; i >= 0; i--)
            {
                var layer = Layers[i].Neurons;
                var nextLayer = Layers[i + 1].Neurons;

                // Errors for the previous layer.
                errors[i] = new double[layer.Length];
                
                for (int c = 0; c < layer.Length; c++)
                {
                    double sum = 0.0;
                    for (int n = 0; n < nextLayer.Length; n++)
                    {
                        double w = nextLayer[n].Weights[c];
                        double e = errors[i + 1][n];
                        sum += w * e;
                    }
                    errors[i][c] = sum;
                }
            }

            // Correcting weights in first layer and bias.
            for (int n = 0; n < Layers[0].Neurons.Length; n++)
            {
                var neuron = Layers[0].Neurons[n];
                double d = ActivationFunctions.SigmoidDerivated(neuron.Output);
                double sigma = errors[0][n] * d;
                for (int w = 0; w < neuron.Weights.Length; w++)
                {
                    double diff = input[n] * sigma * learningRate;
                    neuron.Weights[w] += diff;
                }
                //neuron.Bias -= sigma * learningRate;
            }

            // Correcting other weights.
            for (int i = 1; i < layerCount; i++)
            {
                var layer = Layers[i].Neurons;
                var previousLayer = Layers[i - 1].Neurons;
                for (int n = 0; n < layer.Length; n++)
                {
                    var neuron = layer[n];
                    double d = ActivationFunctions.SigmoidDerivated(neuron.Output);
                    double sigma = errors[i][n] * d;
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        double diff = previousLayer[w].Output * sigma * learningRate;
                        neuron.Weights[w] += diff;
                    }
                    //neuron.Bias -= sigma * learningRate;
                }
            }

            // Update bias.
            //for (int i = 0; i < layerCount; i++)
            //{
            //    for (int n = 0; n < Layers[i].Neurons.Length; n++)
            //    {
            //        Layers[i].Neurons[n].Bias = 0.0;
            //    }
            //}
        }

        public void Train(TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError = true)
        {
            double error = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = maxEpoch / 100;
            double[,] outputs = new double[rowCountY, columnCountY];
            int epoch = 1;
            do
            {
                epoch++;
                for (int s = 0; s < trainData.Inputs.GetLength(0); s++)
                {
                    double[] input = new double[columnCountX];
                    double[] goalOutputs = new double[columnCountY];
                    for (var c = 0; c < columnCountX; c++)
                    {
                        input[c] = trainData.Inputs[s, c];
                    }
                    for (var c = 0; c < columnCountY; c++)
                    {
                        goalOutputs[c] = trainData.Outputs[s, c];
                    }
                    var currentOutput = FeedForward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs[s, c] = currentOutput[c];
                    }
                    BackPropagation(input, goalOutputs, currentOutput, learningRate);
                }
                error = CommonFunctions.GeneralError(outputs, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("({0}) Error: {1}", epoch, error);
                }
            }
            while (error > targetError && epoch < maxEpoch);
        }
    }
}
