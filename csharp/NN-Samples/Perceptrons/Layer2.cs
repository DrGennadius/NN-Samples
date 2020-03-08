using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons
{
    public class Layer2
    {
        public Neuron2[] Neurons;

        public Layer2(double[][] layerWeights, Random r)
        {
            int neuronCount = layerWeights.GetLength(0);
            Neurons = new Neuron2[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                Neurons[i] = new Neuron2(layerWeights[i], r);
            }
        }

        public Layer2(int numberOfNeurons, int numberOfInputs, Random r)
        {
            Neurons = new Neuron2[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron2(numberOfInputs, r);
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
