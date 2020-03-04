using System;

namespace NN_Samples
{
    public class Neuron
    {
        public double[] Weights;
        public double Output;
        public double Bias;

        public Neuron(int numberOfInputs, Random r)
        {
            Bias = 10 * r.NextDouble() - 5;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = 10 * r.NextDouble() - 5;
            }
        }

        public double FeedForward(double[] input)
        {
            double sum = 1.0;
            //double sum = 0.0;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = ActivationFunctions.Sigmoid(sum);
            return Output;
        }
    }
}
