using NN_Samples;
using System;
using System.Linq;

namespace NN_Samples_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            //ANDTest();
            //ORTest();
            //XORTest();
            //MultiplicationTest();
            SimpleNumbersTest();

            Console.ReadKey();
        }

        static void ANDTest()
        {
            Perceptron perceptron = new Perceptron(2, 5, 1);

            TrainData trainData = TrainData.GenerateANDDate();
            perceptron.Train(trainData, 0.005, 0.2, 10000, false);

            ReadyLogicModelTest(perceptron, "AND");
        }

        static void ORTest()
        {
            Perceptron perceptron = new Perceptron(2, 5, 1);

            TrainData trainData = TrainData.GenerateORDate();
            perceptron.Train(trainData, 0.005, 0.2, 10000, false);

            ReadyLogicModelTest(perceptron, "OR");
        }

        static void XORTest()
        {
            Perceptron perceptron = new Perceptron(2, 5, 1);

            TrainData trainData = TrainData.GenerateXORDate();
            perceptron.Train(trainData, 0.005, 0.2, 10000, false);

            ReadyLogicModelTest(perceptron, "XOR");
        }

        static void MultiplicationTest()
        {
            Perceptron perceptron = new Perceptron(2, 64, 1);

            TrainData trainDataOrigin = TrainData.GenerateMultiplicationDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);

            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);

            Console.WriteLine("\nMultiplication training...");
            perceptron.Train(trainDataNormalized, 0.005, 0.1, 100000);

            Console.WriteLine("\nMultiplication result");
            double[,] multiplicationTestData = new double[,]
            {
                { 2, 2 }, { 4, 4 }, { 2, 6 }, { 8, 8 }, { 9, 9 }, { 9, 3 },
                { 9, 0 }, { 0, 9 }, { 0, 3 }, { 0, 0 }, { 0, 1 }, { 5, 5 },
                { 1.5, 1.5 }, { 1.1, 1.1 }, { 8.9, 8.9 }, { 3.3, 3.3 }, { 0, 5.5 }
            };
            ReadyMultiplicationModelTest(perceptron, multiplicationTestData, xMin, xMax, yMin, yMax);
        }

        static void SimpleNumbersTest()
        {
            Perceptron perceptron = new Perceptron(35, 128, 32, 1);
            
            TrainData trainDataOrigin = TrainData.GenerateSimpleNumbersDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);

            Console.WriteLine("\nSimpleNumbers training...");
            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            perceptron.Train(trainDataNormalized, 0.00001, 0.2, 10000);

            Console.WriteLine("\nSimpleNumbers result");
            double[] row = new double[trainDataOrigin.Inputs.GetLength(1)];
            for (int i = 0; i < trainDataOrigin.Inputs.GetLength(0); i++)
            {
                for (int c = 0; c < trainDataOrigin.Inputs.GetLongLength(1); c++)
                {
                    row[c] = trainDataOrigin.Inputs[i, c];
                }
                var output = perceptron.FeedForward(row);
                var denormalizedOutput = CommonFunctions.Denormalize(output, yMin, yMax);
                Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("test {0}: {1}", i, Math.Round(y)))));
            }
        }

        static void ReadyMultiplicationModelTest(Perceptron perceptron, double[,] multiplicationTestData, double xMin, double xMax, double yMin, double yMax)
        {
            for (int i = 0; i < multiplicationTestData.GetLength(0); i++)
            {
                ReadyMultiplicationModelTest(perceptron, multiplicationTestData[i, 0], multiplicationTestData[i, 1], xMin, xMax, yMin, yMax);
            }
        }

        static void ReadyMultiplicationModelTest(Perceptron perceptron, double x1, double x2, double xMin, double xMax, double yMin, double yMax)
        {
            var output = perceptron.FeedForward(new double[] { CommonFunctions.Normalize(x1, xMin, xMax), CommonFunctions.Normalize(x2, xMin, xMax) });
            var denormalizedOutput = CommonFunctions.Denormalize(output, yMin, yMax);
            Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("{0} * {1} = {2}", x1, x2, y))));
        }

        static void ReadyLogicModelTest(Perceptron perceptron, string title)
        {
            Console.WriteLine("\n" + title);

            var output = perceptron.FeedForward(new double[] { 0, 0 });
            Console.WriteLine("0 0 = " + string.Join(" ", output.Select(x => Math.Round(x))));

            output = perceptron.FeedForward(new double[] { 0, 1 });
            Console.WriteLine("0 1 = " + string.Join(" ", output.Select(x => Math.Round(x))));

            output = perceptron.FeedForward(new double[] { 1, 0 });
            Console.WriteLine("1 0 = " + string.Join(" ", output.Select(x => Math.Round(x))));

            output = perceptron.FeedForward(new double[] { 1, 1 });
            Console.WriteLine("1 1 = " + string.Join(" ", output.Select(x => Math.Round(x))));
        }
    }
}
