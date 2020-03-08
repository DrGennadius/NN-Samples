using NN_Samples;
using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives.Samples;
using System;
using System.Linq;

namespace NN_Samples_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            ANDTest();
            NANDTest();
            ORTest();
            XORTest();
            MultiplicationTest();
            SimpleNumbersTest();

            RunAlternativeVariant1();

            Console.ReadKey();
        }

        static void ANDTest()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (AND)");
            ANDTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (AND)");
            ANDTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (AND)");
            ANDTest(perceptron3);
        }

        static void NANDTest()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (NAND)");
            NANDTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (NAND)");
            NANDTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (NAND)");
            NANDTest(perceptron3);
        }

        static void ORTest()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (OR)");
            ORTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (OR)");
            ORTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (OR)");
            ORTest(perceptron3);
        }
        
        static void XORTest()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (XOR)");
            XORTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (XOR)");
            XORTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (XOR)");
            XORTest(perceptron3);
        }

        static void MultiplicationTest()
        {
            var perceptron1 = new SimplePerceptron(2, 64, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (Multiplication)");
            MultiplicationTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (Multiplication)");
            MultiplicationTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (Multiplication)");
            MultiplicationTest(perceptron3);
        }

        static void SimpleNumbersTest()
        {
            var perceptron1 = new SimplePerceptron(35, 128, 32, 1);
            var perceptron2 = new Perceptron(perceptron1);
            var perceptron3 = new Perceptron2(perceptron1, 0.5);

            Console.WriteLine("\nSimple Perceptron (Simple Numbers)");
            SimpleNumbersTest(perceptron1);

            Console.WriteLine("\nPerceptron with bias (Simple Numbers)");
            SimpleNumbersTest(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (Simple Numbers)");
            SimpleNumbersTest(perceptron3);
        }

        static void ANDTest(IPerceptron perceptron)
        {
            TrainData trainData = TrainData.GenerateANDDate();
            TrainStats trainStats = perceptron.Train(trainData, 0.05, 0.2, 400000, false);
            Console.WriteLine(trainStats);

            ReadyLogicModelTest(perceptron);
        }

        static void NANDTest(IPerceptron perceptron)
        {
            TrainData trainData = TrainData.GenerateNANDDate();
            TrainStats trainStats = perceptron.Train(trainData, 0.05, 0.2, 400000, false);
            Console.WriteLine(trainStats);

            ReadyLogicModelTest(perceptron);
        }

        static void ORTest(IPerceptron perceptron)
        {
            TrainData trainData = TrainData.GenerateORDate();
            TrainStats trainStats = perceptron.Train(trainData, 0.05, 0.2, 400000, false);
            Console.WriteLine(trainStats);

            ReadyLogicModelTest(perceptron);
        }

        static void XORTest(IPerceptron perceptron)
        {
            TrainData trainData = TrainData.GenerateXORDate();
            TrainStats trainStats = perceptron.Train(trainData, 0.05, 0.2, 400000, false);
            Console.WriteLine(trainStats);

            ReadyLogicModelTest(perceptron);
        }

        static void MultiplicationTest(IPerceptron perceptron)
        {
            TrainData trainDataOrigin = TrainData.GenerateMultiplicationDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);

            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);

            Console.WriteLine("\nMultiplication training...");
            TrainStats trainStats = perceptron.Train(trainDataNormalized, 0.005, 0.1, 100000, false);
            Console.WriteLine(trainStats);

            Console.WriteLine("\nMultiplication result");
            double[,] multiplicationTestData = new double[,]
            {
                { 2, 2 }, { 4, 4 }, { 2, 6 }, { 8, 8 }, { 9, 9 }, { 9, 3 },
                { 9, 0 }, { 0, 9 }, { 0, 3 }, { 0, 0 }, { 0, 1 }, { 5, 5 },
                { 1.5, 1.5 }, { 1.1, 1.1 }, { 8.9, 8.9 }, { 3.3, 3.3 }, { 0, 5.5 }
            };
            ReadyMultiplicationModelTest(perceptron, multiplicationTestData, xMin, xMax, yMin, yMax);
        }

        static void SimpleNumbersTest(IPerceptron perceptron)
        {
            TrainData trainDataOrigin = TrainData.GenerateSimpleNumbersDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);

            Console.WriteLine("\nSimpleNumbers training...");
            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            TrainStats trainStats = perceptron.Train(trainDataNormalized, 0.00001, 0.2, 10000, false);
            Console.WriteLine(trainStats);

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

        static void ReadyMultiplicationModelTest(IPerceptron perceptron, double[,] multiplicationTestData, double xMin, double xMax, double yMin, double yMax)
        {
            for (int i = 0; i < multiplicationTestData.GetLength(0); i++)
            {
                ReadyMultiplicationModelTest(perceptron, multiplicationTestData[i, 0], multiplicationTestData[i, 1], xMin, xMax, yMin, yMax);
            }
        }

        static void ReadyMultiplicationModelTest(IPerceptron perceptron, double x1, double x2, double xMin, double xMax, double yMin, double yMax)
        {
            var output = perceptron.FeedForward(new double[] { CommonFunctions.Normalize(x1, xMin, xMax), CommonFunctions.Normalize(x2, xMin, xMax) });
            var denormalizedOutput = CommonFunctions.Denormalize(output, yMin, yMax);
            Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("{0} * {1} = {2}", x1, x2, y))));
        }

        static void ReadyLogicModelTest(IPerceptron perceptron)
        {
            var output = perceptron.FeedForward(new double[] { 0, 0 });
            Console.WriteLine("0 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            output = perceptron.FeedForward(new double[] { 0, 1 });
            Console.WriteLine("0 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            output = perceptron.FeedForward(new double[] { 1, 0 });
            Console.WriteLine("1 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            output = perceptron.FeedForward(new double[] { 1, 1 });
            Console.WriteLine("1 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));
        }

        static void RunAlternativeVariant1()
        {
            Console.WriteLine("\nAlternative Variant 1:");
            Console.WriteLine("\nXOR\n{0}", SampleAV1.RunXOR());
            Console.WriteLine("\nAND\n{0}", SampleAV1.RunAND());
            Console.WriteLine("\nOR\n{0}", SampleAV1.RunOR());
        }
    }
}
