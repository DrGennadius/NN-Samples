using NN_Samples.Common;
using NN_Samples.Perceptrons.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Common
{
    public interface IPerceptron : IPerceptronBase
    {
        IList<ILayerBase> Layers { get; set; }

        IPerceptron Add(ILayerBase layer);

        IPerceptron Add(IEnumerable<ILayerBase> layers);

        IPerceptron AddLayer(double[][] layerWeights, double[] layerBias);

        IPerceptron AddLayer(double[][] layerWeights, double[] layerBias, ActivationFunction activationFunction);

        IPerceptron AddLayer(double[][] layerWeights, Random random);

        IPerceptron AddLayer(double[][] layerWeights, ActivationFunction activationFunction, Random random);

        IPerceptron AddLayer(double[,] layerWeights, double[] layerBias);

        IPerceptron AddLayer(double[,] layerWeights, double[] layerBias, ActivationFunction activationFunction);

        IPerceptron AddLayer(double[,] layerWeights, Random random);

        IPerceptron AddLayer(double[,] layerWeights, ActivationFunction activationFunction, Random random);

        IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, Random random);

        IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, ActivationFunction activationFunction, Random random);

        IPerceptron Build();
    }
}
