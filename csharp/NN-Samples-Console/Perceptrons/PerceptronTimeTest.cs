using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples_Console.Perceptrons
{
    public class PerceptronTimeTest
    {
        public static void Test()
        {
            var perceptronTrainer = new PerceptronTrainer();

            var perceptronOld = new PerceptronOld(new int[] { 35, 128, 32, 1 }, new Random(62784123));
            PerceptronTopology perceptronTopology = new PerceptronTopology(new int[] { 2, 5, 1 }, new ActivationFunction(ActivationFunctionType.Sigmoid));
            var perceptron = new Perceptron(perceptronOld);
            var perceptronBase = new PerceptronBase(perceptron);

            var watch = System.Diagnostics.Stopwatch.StartNew();
            TrainStats singleTrainStats = perceptronTrainer.Train(perceptronOld, TrainData.GenerateDataSimpleNumbers(), 0.5, 1e-10, 250000, false, 10);
            Console.WriteLine(singleTrainStats);
            watch.Stop();
            Console.WriteLine($"Old Perceptron Execution Time: {watch.ElapsedMilliseconds} ms");

            watch.Restart();
            singleTrainStats = perceptronTrainer.Train(perceptronBase, TrainData.GenerateDataSimpleNumbers(), 0.5, 1e-10, 250000, false, 10);
            Console.WriteLine(singleTrainStats);
            Console.WriteLine($"Base Perceptron Execution Time: {watch.ElapsedMilliseconds} ms");

            watch.Restart();
            singleTrainStats = perceptronTrainer.Train(perceptron, TrainData.GenerateDataSimpleNumbers(), 0.5, 1e-10, 250000, false, 10);
            Console.WriteLine(singleTrainStats);
            Console.WriteLine($"Perceptron Execution Time: {watch.ElapsedMilliseconds} ms");
        }
    }
}
