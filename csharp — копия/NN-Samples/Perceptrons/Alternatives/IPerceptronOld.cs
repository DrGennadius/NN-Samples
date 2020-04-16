using NN_Samples.Common;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Alternatives
{
    public interface IPerceptronOld : IPerceptronBase
    {
        void TransferWeightsFrom(IPerceptronOld otherPerceptron);

        double[][][] GetWeights();

        void SetWeights(double[][][] weights);

        double[][] GetBiases();

        ActivationFunction GetActivationFunction();

        double GetMomentumRate();
    }
}
