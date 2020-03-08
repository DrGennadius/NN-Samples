using NN_Samples.Common;
using System;

namespace NN_Samples.Perceptrons
{
    public class SimpleNeuron
    {
        public double[] Weights;
        public double Output;
        public double DerivatedOutput;

        public SimpleNeuron(int numberOfInputs, Random r)
        {
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = 10 * r.NextDouble() - 5;
            }
        }

        public double FeedForward(double[] input)
        {
            double sum = 0.0;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = ActivationFunctions.Sigmoid(sum);
            DerivatedOutput = ActivationFunctions.SigmoidDerivated(Output);
            return Output;
        }
    }
}
