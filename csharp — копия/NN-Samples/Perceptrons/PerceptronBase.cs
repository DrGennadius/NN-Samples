using NN_Samples.Common;
using NN_Samples.Perceptrons.Alternatives;
using NN_Samples.Perceptrons.Common;
using NN_Samples.Perceptrons.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class PerceptronBase : IPerceptronBase
    {
        private double[] _input;

        public LayerStruct[] Layers;

        public double[][,] Weights
        {
            get
            {
                double[][,] weights = new double[Layers.Length][,];
                for (int i = 0; i < Layers.Length; i++)
                {
                    weights[i] = Layers[i].Weights;
                }
                return weights;
            }
            set
            {
                if (value.Length == Layers.Length)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        if (value[i].GetLength(0) != Layers[i].Weights.GetLength(0)
                            && value[i].GetLength(1) != Layers[i].Weights.GetLength(1))
                        {
                            throw new Exception("Different the sizes.");
                        }
                        Layers[i].Weights = value[i];
                    }
                }
                else
                {
                    throw new Exception("Different the sizes.");
                }
            }
        }

        public double[][] Biases
        {
            get
            {
                double[][] biases = new double[Layers.Length][];
                for (int i = 0; i < Layers.Length; i++)
                {
                    biases[i] = Layers[i].Biases;
                }
                return biases;
            }
            set
            {
                if (value.Length == Layers.Length)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        if (value[i].Length != Layers[i].Biases.Length)
                        {
                            throw new Exception("Different the sizes.");
                        }
                        Layers[i].Biases = value[i];
                    }
                }
                else
                {
                    throw new Exception("Different the sizes.");
                }
            }
        }

        public double MomentumRate { get; set; }
        public PerceptronTopology Topology { get; set; }

        public PerceptronBase()
        {
        }

        public PerceptronBase(PerceptronTopology topology)
        {
            Topology = topology;
            Initialize();
        }

        public PerceptronBase(PerceptronTopology topology, double momentumRate = 0.5)
        {
            Topology = topology;
            MomentumRate = momentumRate;
            Initialize();
        }

        public PerceptronBase(IPerceptronBase perceptron)
        {
            TransferFrom(perceptron);
        }

        public PerceptronBase(IPerceptronOld perceptronOld)
        {
            TransferFrom(perceptronOld);
        }

        public void Backward(double[] input, double[] targetOutput, double alpha)
        {
            int lastLayerIndex = Topology.GetLayerCount() - 1;

            // From end.
            for (int i = 0; i < targetOutput.Length; i++)
            {
                // Difference between goal output and real output elements.
                double e = Layers[lastLayerIndex].Outputs[i] - targetOutput[i];
                Layers[lastLayerIndex].Deltas[i] = e * Layers[lastLayerIndex].Derivatives[i];
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = lastLayerIndex; k > 0; k--)
            {
                for (int i = 0; i < Layers[k - 1].Weights.GetLength(0); i++)
                {
                    Layers[k - 1].Deltas[i] = 0.0;
                }
                for (int i = 0; i < Layers[k].Weights.GetLength(0); i++)
                {
                    for (int j = 0; j < Layers[k].Weights.GetLength(1); j++)
                    {
                        double w = Layers[k].Weights[i, j];
                        double d = Layers[k].Deltas[i];
                        Layers[k - 1].Deltas[j] += w * d * Layers[k - 1].Derivatives[j];
                    }
                }

                //int sizew = _weights[k].GetLength(1);
                //for (int i = 0; i < sizew; i++)
                //{
                //    _deltas[k - 1][i] = 0;
                //    int sizen = _weights[k].GetLength(0);
                //    for (int j = 0; j < sizen; j++)
                //    {
                //        _deltas[k - 1][i] += _weights[k][j, i] * _deltas[k][j];
                //    }
                //    _deltas[k - 1][i] *= _derivatives[k - 1][i];
                //}
            }

            // Update weights and biases.
            int sizeN = Layers[0].Weights.GetLength(0);
            for (int n = 0; n < sizeN; n++)
            {
                int sizeW = Layers[0].Weights.GetLength(1);
                for (int w = 0; w < sizeW; w++)
                {
                    Layers[0].PrevWeightChanges[n, w] = -alpha * Layers[0].Deltas[n] * _input[w] + Layers[0].PrevWeightChanges[n, w] * MomentumRate;
                    Layers[0].Weights[n, w] += Layers[0].PrevWeightChanges[n, w];
                }
                Layers[0].PrevBiasChanges[n] = -alpha * Layers[0].Deltas[n] + Layers[0].PrevBiasChanges[n] * MomentumRate;
                Layers[0].Biases[n] += Layers[0].PrevBiasChanges[n];
            }
            for (int i = 1; i < Layers.Length; i++)
            {
                sizeN = Layers[i].Weights.GetLength(0);
                for (int n = 0; n < sizeN; n++)
                {
                    int sizeW = Layers[i].Weights.GetLength(1);
                    for (int w = 0; w < sizeW; w++)
                    {
                        Layers[i].PrevWeightChanges[n, w] = -alpha * Layers[i].Deltas[n] * Layers[i - 1].Outputs[n] + Layers[i].PrevWeightChanges[n, w] * MomentumRate;
                        Layers[i].Weights[n, w] += Layers[i].PrevWeightChanges[n, w];
                    }
                    Layers[i].PrevBiasChanges[n] = -alpha * Layers[i].Deltas[n] + Layers[i].PrevBiasChanges[n] * MomentumRate;
                    Layers[i].Biases[n] += Layers[i].PrevBiasChanges[n];
                }
            }
        }

        public object Clone()
        {
            PerceptronBase perceptronBase = new PerceptronBase();
            perceptronBase.TransferFrom(this);
            return perceptronBase;
        }

        public double[] Forward(double[] input)
        {
            _input = input;
            double[] output = Layers[0].Forward(input);
            for (int i = 1; i < Layers.Length; i++)
            {
                output = Layers[i].Forward(output);
            }
            return output;
        }

        public TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            return Train(new TrainData(inputs, outputs), alpha, targetError, maxEpoch, printError);
        }

        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            PerceptronTrainer trainer = new PerceptronTrainer();
            return trainer.Train(this, trainData, alpha, targetError, maxEpoch, printError);
        }

        public void TransferFrom(IPerceptronBase otherPerceptron)
        {
            Layers = new LayerStruct[otherPerceptron.Weights.Length];
            for (int i = 0; i < otherPerceptron.Weights.Length; i++)
            {
                Layers[i].Weights = new double[otherPerceptron.Weights[i].GetLength(0), otherPerceptron.Weights[i].GetLength(1)];
                for (int n = 0; n < otherPerceptron.Weights[i].GetLength(0); n++)
                {
                    for (int w = 0; w < otherPerceptron.Weights[i].GetLength(1); w++)
                    {
                        Layers[i].Weights[n, w] = otherPerceptron.Weights[i][n, w];
                    }
                }
            }
            for (int i = 0; i < otherPerceptron.Biases.Length; i++)
            {
                Layers[i].Biases = new double[otherPerceptron.Biases[i].Length];
                for (int n = 0; n < otherPerceptron.Biases[i].Length; n++)
                {
                    Layers[i].Biases[n] = otherPerceptron.Biases[i][n];
                }
            }
            Topology = otherPerceptron.Topology;
            MomentumRate = otherPerceptron.MomentumRate;
            int[] sizes = otherPerceptron.Topology.GetSizes();
            InitializeInternalValues(sizes);
        }

        public void TransferFrom(IPerceptronOld perceptronOld)
        {
            double[][][] otherWeights = perceptronOld.GetWeights();
            int[] sizes = new int[otherWeights.Length + 1];
            sizes[0] = otherWeights[0][0].Length;
            for (int i = 0; i < otherWeights.Length; i++)
            {
                sizes[i + 1] = otherWeights[i].Length;
            }
            Topology = new PerceptronTopology(sizes, perceptronOld.GetActivationFunction());
            Layers = new LayerStruct[otherWeights.Length];
            double[][] otherBiases = perceptronOld.GetBiases();
            bool haveBiases = otherBiases.Length > 0;
            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i - 1].Weights = new double[sizes[i], sizes[i - 1]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        Layers[i - 1].Weights[n, w] = otherWeights[i - 1][n][w];
                    }
                }
                if (!haveBiases)
                {
                    Layers[i - 1].Biases = new double[sizes[i]];
                }
            }
            MomentumRate = perceptronOld.GetMomentumRate();
            InitializeInternalValues(sizes);
        }

        public void TransferTo(IPerceptronBase otherPerceptron)
        {
            otherPerceptron.TransferFrom(this);
        }

        private void Initialize()
        {
            Random random = new Random(DateTime.Now.Millisecond);

            int[] sizes = Topology.GetSizes();
            int layerCount = sizes.Length - 1;

            Layers = new LayerStruct[layerCount];

            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i - 1].Weights = new double[sizes[i], sizes[i - 1]];
                Layers[i - 1].Biases = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        Layers[i - 1].Weights[n, w] = random.NextDouble() - 0.5;
                    }
                    Layers[i - 1].Biases[n] = random.NextDouble() - 0.5;
                }
            }

            InitializeInternalValues(sizes);
        }

        private void InitializeInternalValues(int[] sizes)
        {
            int layerCount = sizes.Length - 1;

            for (int i = 1; i < sizes.Length; i++)
            {
                Layers[i - 1].ActivationFunction = Topology.ActivationFunctions[i - 1];
                Layers[i - 1].PrevWeightChanges = new double[sizes[i], sizes[i - 1]];
                Layers[i - 1].PrevBiasChanges = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        Layers[i - 1].PrevWeightChanges[n, w] = 0;
                    }
                    Layers[i - 1].PrevBiasChanges[n] = 0;
                }
                Layers[i - 1].Outputs = new double[sizes[i]];
                Layers[i - 1].Derivatives = new double[sizes[i]];
                Layers[i - 1].Deltas = new double[sizes[i]];
            }
        }

        private double[] ForwardLayer(int layerIndex, double[] input)
        {
            double[] result = new double[Layers[layerIndex].Weights.GetLength(0)];
            for (int i = 0; i < Layers[layerIndex].Weights.GetLength(0); i++)
            {
                result[i] = ForwardNeuron(layerIndex, i, input);
            }
            Layers[layerIndex].Outputs = result;
            return result;
        }

        private double ForwardNeuron(int layerIndex, int neuronIndex, double[] input)
        {
            double sum = Layers[layerIndex].Biases[neuronIndex];

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Layers[layerIndex].Weights[neuronIndex, i] * input[i];
            }

            double result = Topology.ActivationFunctions[layerIndex].Calculate(sum);
            Layers[layerIndex].Outputs[neuronIndex] = result;
            Layers[layerIndex].Derivatives[neuronIndex] = Topology.ActivationFunctions[layerIndex].CalculateDerivative(result);
            return result;
        }
    }
}
