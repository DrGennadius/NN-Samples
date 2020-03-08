using NN_Samples.Common;
using System;

namespace NN_Samples.Perceptrons
{
    public class Neuron2
    {
        public double[] Weights;
        public double[] PreviousChanges;
        public double Output;
        public double DerivatedOutput;
        public double Bias;

        public Neuron2(double[] neuronWeights, Random r)
        {
            Bias = 10 * r.NextDouble() - 5;
            Weights = neuronWeights;
            PreviousChanges = new double[neuronWeights.Length];
            for (int i = 0; i < neuronWeights.Length; i++)
            {
                PreviousChanges[i] = 0;
            }
        }

        public Neuron2(int numberOfInputs, Random r)
        {
            Bias = 10 * r.NextDouble() - 5;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = 10 * r.NextDouble() - 5;
            }
            PreviousChanges = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                PreviousChanges[i] = 0;
            }
        }

        public double FeedForward(double[] input)
        {
            double sum = Bias;

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
