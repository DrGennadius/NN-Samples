using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples
{
    public class Layer
    {
        public Neuron[] Neurons;

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
