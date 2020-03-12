using NN_Samples.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Common
{
    public interface IPerceptron
    {
        double[] FeedForward(double[] input);

        void BackPropagation(double[] input, double[] targetOutput, double[] realOutput, double learningRate);

        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false);

        void TransferWeightsFrom(IPerceptron otherPerceptron);

        double[][][] GetWeights();

        void SetWeights(double[][][] weights);
    }
}
