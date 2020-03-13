using NN_Samples.Common;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    /// <summary>
    /// Perceptron with bias and momentum.
    /// </summary>
    public class Perceptron : IPerceptron
    {
        private Layer[] Layers;

        public double MomentumRate = 0.5;

        public ActivationFunction ActivationFunction;

        /// <summary>
        /// Create perceptron with bias and momentum by configuration.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        public Perceptron(params int[] neuronsPerLayer)
            : this(neuronsPerLayer, ActivationFunctionType.Sigmoid)
        {
        }

        /// <summary>
        /// Create perceptron with bias and momentum by configuration, setting activation function and momentum rate.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        /// <param name="activationFunctionType">Activation Function.</param>
        /// <param name="momentumRate">Momentum rate.</param>
        public Perceptron(int[] neuronsPerLayer, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid, double momentumRate = 0.5)
            : this(neuronsPerLayer, new Random(), activationFunctionType, momentumRate)
        {
        }

        /// <summary>
        /// Create perceptron with bias and momentum by configuration, setting activation function, momentum rate and Random.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        /// <param name="random">Random.</param>
        /// <param name="activationFunctionType">Activation Function.</param>
        /// <param name="momentumRate">Momentum rate.</param>
        public Perceptron(int[] neuronsPerLayer, Random random, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid, double momentumRate = 0.5)
            : this(neuronsPerLayer, random, new ActivationFunction(activationFunctionType), momentumRate)
        {
        }

        /// <summary>
        /// Create perceptron with bias and momentum by configuration, setting activation function, momentum rate and Random.
        /// </summary>
        /// <param name="neuronsPerLayer">Configuration.</param>
        /// <param name="random">Random.</param>
        /// <param name="activationFunctionType">Activation Function.</param>
        /// <param name="momentumRate">Momentum rate.</param>
        public Perceptron(int[] neuronsPerLayer, Random random, ActivationFunction activationFunction, double momentumRate = 0.5)
        {
            ActivationFunction = activationFunction;
            Layers = new Layer[neuronsPerLayer.Length - 1];
            MomentumRate = momentumRate;

            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layers[i - 1] = new Layer(neuronsPerLayer[i], neuronsPerLayer[i - 1], random);
            }
        }

        /// <summary>
        /// Create perceptron with bias and momentum by other Perceptron and setting activation function and momentum rate.
        /// </summary>
        /// <param name="perceptron"></param>
        /// <param name="activationFunctionType"></param>
        /// <param name="momentumRate"></param>
        public Perceptron(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid, double momentumRate = 0.5)
            : this(perceptron, new ActivationFunction(activationFunctionType), momentumRate)
        {
        }

        /// <summary>
        /// Create perceptron with bias and momentum by other Perceptron and setting activation function and momentum rate.
        /// </summary>
        /// <param name="perceptron"></param>
        /// <param name="activationFunctionType"></param>
        /// <param name="momentumRate"></param>
        public Perceptron(IPerceptron perceptron, ActivationFunction activationFunction, double momentumRate = 0.5)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            int layerCount = otherWeights.GetLength(0);

            ActivationFunction = activationFunction;
            Layers = new Layer[layerCount];
            MomentumRate = momentumRate;
            Random r = new Random();

            for (int i = 0; i < layerCount; i++)
            {
                Layers[i] = new Layer(otherWeights[i], r);
            }
        }

        public double[] FeedForward(double[] input)
        {
            double[] outputs = new double[0];
            for (int i = 0; i < Layers.Length; i++)
            {
                outputs = Layers[i].FeedForward(ActivationFunction, input);
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
                        double weightChange = - learningRate * deltas[i][n] * neuron.Input[w] + neuron.PreviousChanges[w] * MomentumRate;
                        neuron.Weights[w] += weightChange;
                        neuron.PreviousChanges[w] = weightChange;
                    }
                    double biasChange = -learningRate * deltas[i][n] + neuron.PreviousBiasChange * MomentumRate;
                    neuron.Bias += biasChange;
                    neuron.PreviousBiasChange = biasChange;
                }
            }
        }

        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            PerceptronTrainer perceptronTrainer = new PerceptronTrainer();
            return perceptronTrainer.Train(this, trainData, alpha, targetError, maxEpoch, printError);
        }

        public void TransferWeightsFrom(IPerceptron otherPerceptron)
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
