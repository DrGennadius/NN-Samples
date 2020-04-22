using NN_Samples.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Neurons
{
    public struct Neuron : INeuron
    {
        public double[] Weights { get; set; }

        public double[] Input { get; set; }

        public double[] PreviousChanges { get; set; }

        public double Output { get; set; }

        public double DerivatedOutput { get; set; }

        public double Bias { get; set; }

        public double PreviousBiasChange { get; set; }

        public double Delta { get; set; }

        public Neuron(double[] neuronWeights, double bias)
        {
            Bias = bias;
            Weights = neuronWeights;
            PreviousBiasChange = 0;
            PreviousChanges = new double[neuronWeights.Length];
            for (int i = 0; i < neuronWeights.Length; i++)
            {
                PreviousChanges[i] = 0;
            }
            Input = new double[0];
            Output = 0.0;
            DerivatedOutput = 0.0;
            Delta = 0.0;
        }

        public Neuron(double[] neuronWeights, Random random)
            : this(neuronWeights, random.NextDouble() - 0.5)
        {
        }

        public Neuron(int numberOfInputs, Random random)
        {
            Bias = random.NextDouble() - 0.5;
            Weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = random.NextDouble() - 0.5;
            }
            PreviousBiasChange = 0;
            PreviousChanges = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                PreviousChanges[i] = 0;
            }
            Input = new double[0];
            Output = 0.0;
            DerivatedOutput = 0.0;
            Delta = 0.0;
        }

        public Neuron(INeuron neuron)
        {
            Bias = neuron.Bias;
            Weights = neuron.Weights;
            PreviousChanges = neuron.PreviousChanges;
            PreviousBiasChange = neuron.PreviousBiasChange;
            Delta = neuron.Delta;
            Input = new double[0];
            Output = 0.0;
            DerivatedOutput = 0.0;
        }

        public double Forward(double[] input)
        {
            Input = input;

            double sum = Bias;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = ActivationFunctions.Sigmoid(sum);
            DerivatedOutput = ActivationFunctions.SigmoidDerivated(Output);
            return Output;
        }

        public double Forward(ActivationFunction activationFunction, double[] input)
        {
            Input = input;

            double sum = Bias;

            for (int i = 0; i < Weights.Length; i++)
            {
                sum += Weights[i] * input[i];
            }

            Output = activationFunction.Calculate(sum);
            DerivatedOutput = activationFunction.CalculateDerivative(Output);
            return Output;
        }

        public void Update(double alpha, double momentumRate)
        {
            double neuronAlphaDelta = -alpha * Delta;
            for (int i = 0; i < Weights.Length; i++)
            {
                PreviousChanges[i] = neuronAlphaDelta * Input[i] + PreviousChanges[i] * momentumRate;
                Weights[i] += PreviousChanges[i];
            }
            PreviousBiasChange = neuronAlphaDelta + PreviousBiasChange * momentumRate;
            Bias += PreviousBiasChange;
        }
    }
}
