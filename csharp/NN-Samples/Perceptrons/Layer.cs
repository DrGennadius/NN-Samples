using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class Layer
    {
        public Neuron[] Neurons;

        public Layer(double[][] layerWeights, Random r)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron(layerWeights[i], r);
            }
        }

        public Layer(int numberOfNeurons, int numberOfInputs, Random r)
        {
            Neurons = new Neuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron(numberOfInputs, r);
            }
        }

        public double[] FeedForward(double[] input)
        {
            double[] output = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                output[i] = Neurons[i].FeedForward(input);
            }
            return output;
        }
    }
}
