using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives.Samples;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NN_Samples_Console.Perceptrons
{
    public class PerceptronTypeTest
    {
        public static double LogicLearningRate { get; set; } = 0.5;
        public static double LogicTargetError { get; set; } = 1e-6;
        public static int LogicMaxEpoch { get; set; } = 1000000;
        public static bool LogicPrintError { get; set; } = false;

        public static void Test()
        {
            TestAND();
            TestNAND();
            TestOR();
            TestXOR();
            TestMultiplication();
            TestSimpleNumbers();

            RunAlternativeVariant1();
        }

        public static void TestAND()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new SimplePerceptron2(perceptron1);
            var perceptron3 = new Perceptron(perceptron1);

            Console.WriteLine("\nSimple Perceptron (AND)");
            TestAND(perceptron1);

            Console.WriteLine("\nPerceptron with bias (AND)");
            TestAND(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (AND)");
            TestAND(perceptron3);
        }

        public static void TestNAND()
        {
            var perceptron1 = new SimplePerceptron(2, 1);
            var perceptron2 = new SimplePerceptron2(2, 3, 1);
            var perceptron3 = new Perceptron(2, 3, 1);

            Console.WriteLine("\nSimple Perceptron (NAND)");
            TestNAND(perceptron1);

            Console.WriteLine("\nPerceptron with bias (NAND)");
            TestNAND(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (NAND)");
            TestNAND(perceptron3);
        }

        public static void TestOR()
        {
            var perceptron1 = new SimplePerceptron(2, 3, 1);
            var perceptron2 = new SimplePerceptron2(2, 3, 1);
            var perceptron3 = new Perceptron(2, 3, 1);

            Console.WriteLine("\nSimple Perceptron (OR)");
            TestOR(perceptron1);

            Console.WriteLine("\nPerceptron with bias (OR)");
            TestOR(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (OR)");
            TestOR(perceptron3);
        }

        public static void TestXOR()
        {
            var perceptron1 = new SimplePerceptron(2, 3, 1);
            var perceptron2 = new SimplePerceptron2(2, 3, 1);
            var perceptron3 = new Perceptron(2, 3, 1);

            Console.WriteLine("\nSimple Perceptron (XOR)");
            TestXOR(perceptron1);

            Console.WriteLine("\nPerceptron with bias (XOR)");
            TestXOR(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (XOR)");
            TestXOR(perceptron3);
        }

        public static void TestMultiplication()
        {
            var perceptron1 = new SimplePerceptron(2, 64, 1);
            var perceptron2 = new SimplePerceptron2(perceptron1);
            var perceptron3 = new Perceptron(perceptron1);

            Console.WriteLine("\nSimple Perceptron (Multiplication)");
            TestMultiplication(perceptron1);

            Console.WriteLine("\nPerceptron with bias (Multiplication)");
            TestMultiplication(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (Multiplication)");
            TestMultiplication(perceptron3);
        }

        public static void TestSimpleNumbers()
        {
            var perceptron1 = new SimplePerceptron(35, 128, 32, 1);
            var perceptron2 = new SimplePerceptron2(perceptron1);
            var perceptron3 = new Perceptron(perceptron1);

            Console.WriteLine("\nSimple Perceptron (Simple Numbers)");
            TestSimpleNumbers(perceptron1);

            Console.WriteLine("\nPerceptron with bias (Simple Numbers)");
            TestSimpleNumbers(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (Simple Numbers)");
            TestSimpleNumbers(perceptron3);
        }

        public static void TestAND(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateANDDate(), activationFunctionType);
        }

        public static void TestNAND(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateNANDDate(), activationFunctionType);
        }

        public static void TestOR(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateORDate(), activationFunctionType);
        }

        public static void TestXOR(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateXORDate(), activationFunctionType);
        }

        public static void TestLogic(IPerceptron perceptron, TrainData trainData, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainData, LogicTargetError, LogicLearningRate, LogicMaxEpoch, LogicPrintError);
            Console.WriteLine(trainStats);

            TestReadyLogicModel(perceptron, activationFunctionType);
        }

        public static void TestMultiplication(IPerceptron perceptron)
        {
            TrainData trainDataOrigin = TrainData.GenerateMultiplicationDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);

            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);

            Console.WriteLine("\nMultiplication training...");
            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainDataNormalized, 0.0005, 0.5, 100000, false);
            Console.WriteLine(trainStats);

            Console.WriteLine("\nMultiplication result");
            double[,] multiplicationTestData = new double[,]
            {
                { 2, 2 }, { 4, 4 }, { 2, 6 }, { 8, 8 }, { 9, 9 }, { 9, 3 },
                { 9, 0 }, { 0, 9 }, { 0, 3 }, { 0, 0 }, { 0, 1 }, { 5, 5 },
                { 1.5, 1.5 }, { 1.1, 1.1 }, { 8.9, 8.9 }, { 3.3, 3.3 }, { 0, 5.5 }
            };
            TestReadyMultiplicationModel(perceptron, multiplicationTestData, xMin, xMax, yMin, yMax);
        }

        public static void TestSimpleNumbers(IPerceptron perceptron)
        {
            TrainData trainDataOrigin = TrainData.GenerateSimpleNumbersDate();
            TrainData trainDataNormalized = CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);

            Console.WriteLine("\nSimpleNumbers training...");
            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainDataNormalized, 0.001, 0.5, 100000, false);
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

        public static void TestReadyMultiplicationModel(IPerceptron perceptron, double[,] multiplicationTestData, double xMin, double xMax, double yMin, double yMax)
        {
            for (int i = 0; i < multiplicationTestData.GetLength(0); i++)
            {
                TestReadyMultiplicationModel(perceptron, multiplicationTestData[i, 0], multiplicationTestData[i, 1], xMin, xMax, yMin, yMax);
            }
        }

        public static void TestReadyMultiplicationModel(IPerceptron perceptron, double x1, double x2, double xMin, double xMax, double yMin, double yMax)
        {
            var output = perceptron.FeedForward(new double[] { CommonFunctions.Normalize(x1, xMin, xMax), CommonFunctions.Normalize(x2, xMin, xMax) });
            var denormalizedOutput = CommonFunctions.Denormalize(output, yMin, yMax);
            Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("{0} * {1} = {2}", x1, x2, y))));
        }

        public static void TestReadyLogicModel(IPerceptron perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            var sample = activationFunctionType == ActivationFunctionType.HyperbolicTangent ? 
                new double[] { -1, -1 } : new double[] { 0, 0 };
            var output = perceptron.FeedForward(sample);
            Console.WriteLine("0 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            sample = activationFunctionType == ActivationFunctionType.HyperbolicTangent ?
                new double[] { -1, 1 } : new double[] { 0, 1 };
            output = perceptron.FeedForward(sample);
            Console.WriteLine("0 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            sample = activationFunctionType == ActivationFunctionType.HyperbolicTangent ?
                new double[] { 1, -1 } : new double[] { 1, 0 };
            output = perceptron.FeedForward(sample);
            Console.WriteLine("1 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

            sample = new double[] { 1, 1 };
            output = perceptron.FeedForward(sample);
            Console.WriteLine("1 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));
        }

        public static void RunAlternativeVariant1()
        {
            Console.WriteLine("\nAlternative Variant 1:");
            Console.WriteLine("\nComparison\n{0}", SampleAV1.Comparison());
            Console.WriteLine("\nXOR\n{0}", SampleAV1.RunXOR());
            Console.WriteLine("\nAND\n{0}", SampleAV1.RunAND());
            Console.WriteLine("\nNAND\n{0}", SampleAV1.RunNAND());
            Console.WriteLine("\nOR\n{0}", SampleAV1.RunOR());
        }
    }
}
