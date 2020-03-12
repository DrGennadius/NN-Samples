using NN_Samples.Common;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    /// <summary>
    /// Perceptron with bias.
    /// </summary>
    public class SimplePerceptron2 : IPerceptron
    {
        SimpleLayer2[] Layers;

        public SimplePerceptron2(params int[] neuronsPerLayer)
            : this(neuronsPerLayer, new Random())
        {
        }

        public SimplePerceptron2(int[] neuronsPerLayer, Random random)
        {
            Layers = new SimpleLayer2[neuronsPerLayer.Length - 1];

            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layers[i - 1] = new SimpleLayer2(neuronsPerLayer[i], neuronsPerLayer[i - 1], random);
            }
        }

        public SimplePerceptron2(IPerceptron perceptron)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            int layerCount = otherWeights.GetLength(0);

            Layers = new SimpleLayer2[layerCount];
            Random r = new Random();

            for (int i = 0; i < layerCount; i++)
            {
                Layers[i] = new SimpleLayer2(otherWeights[i], r);
            }
        }

        public double[] FeedForward(double[] input)
        {
            double[] outputs = new double[0];
            for (int i = 0; i < Layers.Length; i++)
            {
                outputs = Layers[i].FeedForward(input);
                input = outputs;
            }
            return outputs;
        }

        public void BackPropagation(double[] input, double[] targetOutput, double[] realOutput, double learningRate)
        {
            double[][] deltas = new double[Layers.Length][];
            int lastLayerIndex = Layers.Length - 1;

            // From end.
            deltas[lastLayerIndex] = new double[Layers[lastLayerIndex].Neurons.Length];
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                double e = Layers[lastLayerIndex].Neurons[i].Output - targetOutput[i];
                deltas[lastLayerIndex][i] = e * Layers[lastLayerIndex].Neurons[i].DerivatedOutput;
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = lastLayerIndex; k > 0; k--)
            {
                var layer = Layers[k].Neurons;
                var previousLayer = Layers[k - 1].Neurons;

                deltas[k - 1] = new double[previousLayer.Length];

                for (int i = 0; i < previousLayer.Length; i++)
                {
                    deltas[k - 1][i] = 0.0;
                }
                for (int i = 0; i < layer.Length; i++)
                {
                    for (int j = 0; j < layer[i].Weights.Length; j++)
                    {
                        double w = layer[i].Weights[j];
                        double d = deltas[k][i];
                        deltas[k - 1][j] += w * d * previousLayer[j].DerivatedOutput;
                    }
                }

                //for (int i = 0; i < layer[0].Weights.Length; i++)
                //{
                //    deltas[k - 1][i] = 0;
                //    for (int j = 0; j < layer.Length; j++)
                //    {
                //        deltas[k - 1][i] += Layers[k].Neurons[j].Weights[i] * deltas[k][j];
                //    }
                //    deltas[k - 1][i] *= Layers[k - 1].Neurons[i].DerivatedOutput;
                //}
            }

            // Correcting weights and bias.
            for (int i = 0; i < lastLayerIndex + 1; i++)
            {
                var layer = Layers[i].Neurons;
                for (int n = 0; n < layer.Length; n++)
                {
                    var neuron = layer[n];
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        neuron.Weights[w] -= learningRate * deltas[i][n] * neuron.Input[w];
                        neuron.Bias -= learningRate * deltas[i][n];
                    }
                }
            }
        }

        public TrainStats Train(TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError = true)
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
            while (error > targetError && epoch < maxEpoch);
            TrainStats trainStats = new TrainStats
            {
                LastError = error,
                NumberOfEpoch = epoch
            };
            return trainStats;
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
            int otherLayerSize = weights.Length;
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
