using NN_Samples.Common;
using NN_Samples.Perceptrons.Alternatives;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class PerceptronBase : IPerceptronBase
    {
        private double[][,] _weights;
        private double[][,] _prevWeightChanges;
        private double[][] _biases;
        private double[][] _biasChanges;
        private double[][] _outputs;
        private double[][] _derivatives;
        private double[][] _deltas;
        private double[] _input;

        public double[][,] Weights { get => _weights; set => _weights = value; }
        public double[][] Biases { get => _biases; set => _biases = value; }

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
                double e = _outputs[lastLayerIndex][i] - targetOutput[i];
                _deltas[lastLayerIndex][i] = e * _derivatives[lastLayerIndex][i];
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = lastLayerIndex; k > 0; k--)
            {
                var prevLayerDeltas = _deltas[k - 1];
                var currentLayerDeltas = _deltas[k];
                var prevLayerWeights = _weights[k - 1];
                var currentLayerWeights = _weights[k];
                var prevLayerDerivatives = _derivatives[k - 1];
                int sizen = prevLayerWeights.GetLength(0);
                for (int i = 0; i < sizen; i++)
                {
                    prevLayerDeltas[i] = 0.0;
                }
                //int sizen = _weights[k].GetLength(0);
                //for (int i = 0; i < sizen; i++)
                //{
                //    int sizew = _weights[k].GetLength(1);
                //    for (int j = 0; j < sizew; j++)
                //    {
                //        double w = _weights[k][i, j];
                //        double d = _deltas[k][i];
                //        _deltas[k - 1][j] += w * d * _derivatives[k - 1][j];
                //    }
                //}

                int sizew = currentLayerWeights.GetLength(1);
                for (int i = 0; i < sizew; i++)
                {
                    double prevDelta = 0;
                    sizen = currentLayerWeights.GetLength(0);
                    for (int j = 0; j < sizen; j++)
                    {
                        prevDelta += currentLayerWeights[j, i] * currentLayerDeltas[j];
                    }
                    prevLayerDeltas[i] = prevDelta * prevLayerDerivatives[i];
                }
            }

            // Update weights and biases.
            var layerWeights = _weights[0];
            var layerBiases = _biases[0];
            var layerDeltas = _deltas[0];
            var layerPrevWeightChanges = _prevWeightChanges[0];
            int sizeN = layerWeights.GetLength(0);
            for (int n = 0; n < sizeN; n++)
            {
                var alphaDelta = -alpha * layerDeltas[n];
                int sizeW = layerWeights.GetLength(1);
                for (int w = 0; w < sizeW; w++)
                {
                    double weightChange = alphaDelta * _input[w] + layerPrevWeightChanges[n, w] * MomentumRate;
                    layerPrevWeightChanges[n, w] = weightChange;
                    layerWeights[n, w] += weightChange;
                }
                double biasChange = alphaDelta + _biasChanges[0][n] * MomentumRate;
                _biasChanges[0][n] = biasChange;
                layerBiases[n] += biasChange;
            }
            int sizeL = _weights.Length;
            for (int i = 1; i < sizeL; i++)
            {
                var prevOutputs = _outputs[i - 1];
                layerWeights = _weights[i];
                layerBiases = _biases[i];
                layerDeltas = _deltas[i];
                layerPrevWeightChanges = _prevWeightChanges[i];
                sizeN = layerWeights.GetLength(0);
                for (int n = 0; n < sizeN; n++)
                {
                    var alphaDelta = -alpha * layerDeltas[n];
                    int sizeW = layerWeights.GetLength(1);
                    for (int w = 0; w < sizeW; w++)
                    {
                        double weightChange = alphaDelta * prevOutputs[w] + layerPrevWeightChanges[n, w] * MomentumRate;
                        _prevWeightChanges[i][n, w] = weightChange;
                        layerWeights[n, w] += weightChange;
                    }
                    double biasChange = alphaDelta + _biasChanges[i][n] * MomentumRate;
                    _biasChanges[i][n] = biasChange;
                    layerBiases[n] += biasChange;
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
            int sizeL = _weights.Length;
            for (int i = 0; i < sizeL; i++)
            {
                var activationFunction = Topology.ActivationFunctions[i].Calculate;
                var derivativeActivationFunction = Topology.ActivationFunctions[i].CalculateDerivative;
                var layerOutputs = _outputs[i];
                var layerDerivatives = _derivatives[i];
                var layerBiases = _biases[i];
                var layerWeights = _weights[i];
                int sizeN = layerWeights.GetLength(0);
                for (int n = 0; n < sizeN; n++)
                {
                    double sum = layerBiases[n];
                    int sizeW = layerWeights.GetLength(1);
                    for (int w = 0; w < sizeW; w++)
                    {
                        sum += layerWeights[n, w] * input[w];
                    }
                    layerOutputs[n] = activationFunction(sum);
                    layerDerivatives[n] = derivativeActivationFunction(layerOutputs[n]);
                }
                input = layerOutputs;
            }
            return input;
        }

        public TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false)
        {
            return Train(new TrainData(inputs, outputs), alpha, targetError, maxEpoch, printError);
        }

        public TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false)
        {

            throw new NotImplementedException();
        }

        public void TransferFrom(IPerceptronBase otherPerceptron)
        {
            _weights = new double[otherPerceptron.Weights.Length][,];
            for (int i = 0; i < otherPerceptron.Weights.Length; i++)
            {
                _weights[i] = new double[otherPerceptron.Weights[i].GetLength(0), otherPerceptron.Weights[i].GetLength(1)];
                for (int n = 0; n < otherPerceptron.Weights[i].GetLength(0); n++)
                {
                    for (int w = 0; w < otherPerceptron.Weights[i].GetLength(1); w++)
                    {
                        _weights[i][n, w] = otherPerceptron.Weights[i][n, w];
                    }
                }
            }
            _biases = new double[otherPerceptron.Biases.Length][];
            for (int i = 0; i < otherPerceptron.Biases.Length; i++)
            {
                _biases[i] = new double[otherPerceptron.Biases[i].Length];
                for (int n = 0; n < otherPerceptron.Biases[i].Length; n++)
                {
                    _biases[i][n] = otherPerceptron.Biases[i][n];
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
            _weights = new double[otherWeights.Length][,];
            double[][] otherBiases = perceptronOld.GetBiases();
            bool haveBiases = otherBiases.Length > 0;
            _biases = haveBiases ? perceptronOld.GetBiases() : new double[otherWeights.Length][];
            for (int i = 1; i < sizes.Length; i++)
            {
                _weights[i - 1] = new double[sizes[i], sizes[i - 1]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _weights[i - 1][n, w] = otherWeights[i - 1][n][w];
                    }
                }
                if (!haveBiases)
                {
                    _biases[i - 1] = new double[sizes[i]];
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

            _weights = new double[layerCount][,];
            _biases = new double[layerCount][];

            for (int i = 1; i < sizes.Length; i++)
            {
                _weights[i - 1] = new double[sizes[i], sizes[i - 1]];
                _biases[i - 1] = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _weights[i - 1][n, w] = random.NextDouble() - 0.5;
                    }
                    _biases[i - 1][n] = random.NextDouble() - 0.5;
                }
            }

            InitializeInternalValues(sizes);
        }

        private void InitializeInternalValues(int[] sizes)
        {
            int layerCount = sizes.Length - 1;

            _prevWeightChanges = new double[layerCount][,];
            _biasChanges = new double[layerCount][];
            _outputs = new double[layerCount][];
            _derivatives = new double[layerCount][];
            _deltas = new double[layerCount][];

            for (int i = 1; i < sizes.Length; i++)
            {
                _prevWeightChanges[i - 1] = new double[sizes[i], sizes[i - 1]];
                _biasChanges[i - 1] = new double[sizes[i]];
                for (int n = 0; n < sizes[i]; n++)
                {
                    for (int w = 0; w < sizes[i - 1]; w++)
                    {
                        _prevWeightChanges[i - 1][n, w] = 0;
                    }
                    _biasChanges[i - 1][n] = 0;
                }
                _outputs[i - 1] = new double[sizes[i]];
                _derivatives[i - 1] = new double[sizes[i]];
                _deltas[i - 1] = new double[sizes[i]];
            }
        }
    }
}
