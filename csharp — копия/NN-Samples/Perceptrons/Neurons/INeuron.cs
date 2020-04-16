using NN_Samples.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Neurons
{
    public interface INeuron
    {
        double[] Weights { get; set; }
        double[] Input { get; set; }
        double[] PreviousChanges { get; set; }
        double Bias { get; set; }
        double Output { get; set; }
        double DerivatedOutput { get; set; }
        double PreviousBiasChange { get; set; }
        double Delta { get; set; }

        double Forward(double[] input);
        double Forward(ActivationFunction activationFunction, double[] input);
    }
}
