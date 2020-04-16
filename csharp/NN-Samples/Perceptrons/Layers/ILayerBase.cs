using NN_Samples.Common;
using NN_Samples.Perceptrons.Neurons;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Layers
{
    public interface ILayerBase
    {
        Neuron[] Neurons { get; set; }

        double[] Output { get; }

        double[,] Weights { get; }

        ActivationFunction ActivationFunction { get; set; }

        double[] Forward(double[] input);

        double[] Forward(ILayerBase layer);
    }
}
