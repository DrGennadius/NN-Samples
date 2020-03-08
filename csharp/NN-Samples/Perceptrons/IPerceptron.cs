using NN_Samples.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public interface IPerceptron
    {
        double[] FeedForward(double[] input);

        void BackPropagation(double[] input, double[] targetOutput, double[] realOutput, double learningRate);

        TrainStats Train(TrainData trainData, double targetError, double learningRate, int maxEpoch, bool printError = true);

        void TransferWeights(IPerceptron otherPerceptron);

        double[][][] GetWeights();

        void SetWeights(double[][][] weights);
    }
}
