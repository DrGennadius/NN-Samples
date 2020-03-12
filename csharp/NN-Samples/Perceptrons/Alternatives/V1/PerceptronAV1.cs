/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using NN_Samples.Common;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Perceptron. Class based on programforyou.ru sample
    /// </summary>
    public class PerceptronAV1
    {
        private struct LayerT
        {
            public VectorAV1 Input; // layer input
            public VectorAV1 Output; // activated layer output
            public VectorAV1 DerivatedOutput; // derivative of layer activation function
        }

        private MatrixAV1[] Weights; // layer weight matrices
        private LayerT[] LayerValues; // values on each layer
        private VectorAV1[] Deltas; // error deltas on each layer

        private int LayerCount; // number of layers

        public PerceptronAV1(int[] sizes)
        {
            Random random = new Random(DateTime.Now.Millisecond);

            LayerCount = sizes.Length - 1;

            Weights = new MatrixAV1[LayerCount];
            LayerValues = new LayerT[LayerCount];
            Deltas = new VectorAV1[LayerCount];

            for (int k = 1; k < sizes.Length; k++)
            {
                Weights[k - 1] = new MatrixAV1(sizes[k], sizes[k - 1], random);

                LayerValues[k - 1].Input = new VectorAV1(sizes[k - 1]);
                LayerValues[k - 1].Output = new VectorAV1(sizes[k]);
                LayerValues[k - 1].DerivatedOutput = new VectorAV1(sizes[k]);

                Deltas[k - 1] = new VectorAV1(sizes[k]);
            }
        }

        public PerceptronAV1(IPerceptron perceptron)
        {
            double[][][] otherWeights = perceptron.GetWeights();
            LayerCount = otherWeights.Length;

            Weights = new MatrixAV1[LayerCount];
            LayerValues = new LayerT[LayerCount];
            Deltas = new VectorAV1[LayerCount];

            int[] sizes = new int[LayerCount + 1];
            sizes[0] = otherWeights[0][0].Length;
            for (int i = 1; i <= LayerCount; i++)
            {
                sizes[i] = otherWeights[i - 1].Length;
            }

            for (int i = 1; i < sizes.Length; i++)
            {
                Weights[i - 1] = new MatrixAV1(otherWeights[i - 1]);

                LayerValues[i - 1].Input = new VectorAV1(sizes[i - 1]);
                LayerValues[i - 1].Output = new VectorAV1(sizes[i]);
                LayerValues[i - 1].DerivatedOutput = new VectorAV1(sizes[i]);

                Deltas[i - 1] = new VectorAV1(sizes[i]);
            }
        }

        public VectorAV1 Forward(VectorAV1 input)
        {
            for (int k = 0; k < LayerCount; k++)
            {
                if (k == 0)
                {
                    for (int i = 0; i < input.Length; i++)
                    {
                        LayerValues[k].Input[i] = input[i];
                    }
                }
                else
                {
                    for (int i = 0; i < LayerValues[k - 1].Output.Length; i++)
                    {
                        LayerValues[k].Input[i] = LayerValues[k - 1].Output[i];
                    }
                }

                for (int i = 0; i < Weights[k].RowSize; i++)
                {
                    double y = 0;

                    for (int j = 0; j < Weights[k].ColumnSize; j++)
                    {
                        y += Weights[k][i, j] * LayerValues[k].Input[j];
                    }

                    // activation by sigmoid function
                    LayerValues[k].Output[i] = 1 / (1 + Math.Exp(-y));
                    LayerValues[k].DerivatedOutput[i] = LayerValues[k].Output[i] * (1 - LayerValues[k].Output[i]);

                    // activation by hyperbolic tangent function
                    //LayerValues[k].Output[i] = Math.Tanh(y);
                    //LayerValues[k].DerivatedOutput[i] = 1 - L[k].z[i] * L[k].z[i];

                    // activation by ReLU function
                    //LayerValues[k].Output[i] = y > 0 ? y : 0;
                    //LayerValues[k].DerivatedOutput[i] = y > 0 ? 1 : 0;
                }
            }

            return LayerValues[LayerCount - 1].Output;
        }

        public void Backward(VectorAV1 output, ref double error)
        {
            int last = LayerCount - 1;

            error = 0;

            for (int i = 0; i < output.Length; i++)
            {
                double e = LayerValues[last].Output[i] - output[i];

                Deltas[last][i] = e * LayerValues[last].DerivatedOutput[i];
                error += e * e / 2;
            }

            // Calculate each previous delta based on the current one
            // by multiplying by the transposed matrix.
            for (int k = last; k > 0; k--)
            {
                for (int i = 0; i < Weights[k].ColumnSize; i++)
                {
                    Deltas[k - 1][i] = 0;

                    for (int j = 0; j < Weights[k].RowSize; j++)
                    {
                        Deltas[k - 1][i] += Weights[k][j, i] * Deltas[k][j];
                    }

                    // multiply the resulting value by the derivative of the previous layer
                    Deltas[k - 1][i] *= LayerValues[k - 1].DerivatedOutput[i];
                }
            }
        }

        /// <summary>
        /// Update weights
        /// </summary>
        /// <param name="alpha">Learning rate</param>
        public void UpdateWeights(double alpha)
        {
            for (int k = 0; k < LayerCount; k++)
            {
                for (int i = 0; i < Weights[k].RowSize; i++)
                {
                    for (int j = 0; j < Weights[k].ColumnSize; j++)
                    {
                        Weights[k][i, j] -= alpha * Deltas[k][i] * LayerValues[k].Input[j];
                    }
                }
            }
        }

        /// <summary>
        /// Train
        /// </summary>
        /// <param name="X">Inputs</param>
        /// <param name="Y">Outputs</param>
        /// <param name="alpha">Learning rate</param>
        /// <param name="eps">Target error</param>
        /// <param name="epochs">Epoch number limit</param>
        public TrainStats Train(VectorAV1[] X, VectorAV1[] Y, double alpha, double eps, int epochs)
        {
            int epoch = 1; // номер эпохи

            double error; // epoch error

            do
            {
                error = 0;

                // Go through all the elements of the training set
                for (int i = 0; i < X.Length; i++)
                {
                    Forward(X[i]);
                    Backward(Y[i], ref error);
                    UpdateWeights(alpha);
                }

                // Console.WriteLine("epoch: {0}, error: {1}", epoch, error);

                epoch++;
            } while (epoch < epochs && error > eps);
            return new TrainStats
            {
                LastError = error,
                NumberOfEpoch = epoch
            };
        }
    }
}
