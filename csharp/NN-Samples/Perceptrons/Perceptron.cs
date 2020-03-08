using NN_Samples.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class Perceptron : IPerceptron
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

        public Perceptron(IPerceptron perceptron)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            int layerCount = otherWeights.GetLength(0);

            Layers = new Layer[layerCount];
            Random r = new Random();

            for (int i = 0; i < layerCount; i++)
            {
                Layers[i] = new Layer(otherWeights[i], r);
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

        public void BackPropagation(double[] input, double[] targetOutput, double[] realOutput, double learningRate)
        {
            double[][] errors = new double[Layers.GetLength(0)][];
            int layerCount = Layers.GetLength(0);

            // From end.
            errors[layerCount - 1] = new double[Layers[layerCount - 1].Neurons.Length];
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                errors[layerCount - 1][i] = targetOutput[i] - realOutput[i];
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
                double sigma = errors[0][n] * neuron.DerivatedOutput;
                for (int w = 0; w < neuron.Weights.Length; w++)
                {
                    double delta = input[n] * sigma;
                    neuron.Weights[w] += delta * learningRate;
                }
                neuron.Bias += sigma * learningRate;
            }

            // Correcting other weights and bias.
            for (int i = 1; i < layerCount; i++)
            {
                var layer = Layers[i].Neurons;
                var previousLayer = Layers[i - 1].Neurons;
                for (int n = 0; n < layer.Length; n++)
                {
                    var neuron = layer[n];
                    double sigma = errors[i][n] * neuron.DerivatedOutput;
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        double delta = previousLayer[w].Output * sigma;
                        neuron.Weights[w] += delta * learningRate;
                    }
                    neuron.Bias += sigma * learningRate;
                }
            }
        }

        public double Train(TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError = true)
        {
            double error = double.MaxValue;
            int rowCountX = trainData.Inputs.GetLength(0);
            int columnCountX = trainData.Inputs.GetLength(1);
            int rowCountY = trainData.Outputs.GetLength(0);
            int columnCountY = trainData.Outputs.GetLength(1);
            int printErrorStep = maxEpoch / 100;
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
                    var currentOutput = FeedForward(input);
                    for (var c = 0; c < columnCountY; c++)
                    {
                        outputs[s, c] = currentOutput[c];
                    }
                    BackPropagation(input, targetOutputs, currentOutput, learningRate);
                }
                error = CommonFunctions.GeneralError(outputs, trainData.Outputs);
                if (printError && epoch % printErrorStep == 0)
                {
                    Console.WriteLine("({0}) Error: {1}", epoch, error);
                }
            }
            while (error > targetError && epoch <= maxEpoch);
            return error;
        }

        public void TransferWeights(IPerceptron otherPerceptron)
        {
            SetWeights(otherPerceptron.GetWeights());
        }

        public double[][][] GetWeights()
        {
            double[][][] weights = new double[Layers.Length][][];
            for (int i = 0; i < Layers.Length; i++)
            {
                weights[i] = new double[Layers[i].Neurons.Length][];
                for (int n = 0; n < Layers[i].Neurons.Length; n++)
                {
                    weights[i][n] = new double[Layers[i].Neurons[n].Weights.Length];
                    for (int w = 0; w < Layers[i].Neurons[n].Weights.Length; w++)
                    {
                        weights[i][n][w] = Layers[i].Neurons[n].Weights[w];
                    }
                }
            }
            return weights;
        }

        public void SetWeights(double[][][] weights)
        {
            int otherLayerSize = weights.GetLength(0);
            if (otherLayerSize != Layers.Length)
            {
                throw new ArgumentException("Sizes of perceptrons are not equal");
            }
            for (int i = 0; i < otherLayerSize; i++)
            {
                int otherNeuronSize = weights[i].GetLength(0);
                if (otherNeuronSize != Layers[i].Neurons.Length)
                {
                    throw new ArgumentException("Sizes of perceptrons are not equal");
                }
                for (int n = 0; n < otherNeuronSize; n++)
                {
                    int otherWeightsSize = weights[i][n].GetLength(0);
                    if (otherWeightsSize != Layers[i].Neurons[n].Weights.Length)
                    {
                        throw new ArgumentException("Sizes of perceptrons are not equal");
                    }
                }
            }
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int n = 0; n < Layers[i].Neurons.Length; n++)
                {
                    for (int w = 0; w < Layers[i].Neurons[n].Weights.Length; w++)
                    {
                        Layers[i].Neurons[n].Weights[w] = weights[i][n][w];
                    }
                }
            }
        }
    }
}
