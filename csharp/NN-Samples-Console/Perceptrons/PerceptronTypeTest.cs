using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives;
using NN_Samples.Perceptrons.Alternatives.Samples;
using NN_Samples.Perceptrons.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

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
            Console.WriteLine("\n#########################");
            Console.WriteLine("#### Perceptron Types ###");
            Console.WriteLine("#########################");

            Comparison();

            var perceptronOld = new PerceptronOld(2, 5, 1);
            var perceptronBase = new PerceptronBase(perceptronOld);
            var perceptron = new Perceptron(perceptronBase);

            TestLogic(perceptronOld, "Perceptron Old");
            TestLogic(perceptronBase, "Perceptron Base");
            TestLogic(perceptron, "Perceptron");

            perceptronOld = new PerceptronOld(2, 64, 1);
            perceptronBase = new PerceptronBase(perceptronOld);
            perceptron = new Perceptron(perceptronBase);
            TestMultiplication(perceptron, TrainData.GenerateDataMultiplication(), "Perceptron Old Multiplication");
            TestMultiplication(perceptron, TrainData.GenerateDataMultiplication(), "Perceptron Base Multiplication");
            TestMultiplication(perceptron, TrainData.GenerateDataMultiplication(), "Perceptron Multiplication");

            perceptronOld = new PerceptronOld(35, 128, 32, 1);
            perceptronBase = new PerceptronBase(perceptronOld);
            perceptron = new Perceptron(perceptronBase);
            TestSimpleNumbers(perceptronOld, TrainData.GenerateDataSimpleNumbers(), "Perceptron Old SimpleNumbers");
            TestSimpleNumbers(perceptronBase, TrainData.GenerateDataSimpleNumbers(), "Perceptron Base SimpleNumbers");
            TestSimpleNumbers(perceptron, TrainData.GenerateDataSimpleNumbers(), "Perceptron SimpleNumbers");

            RunAlternativeVariant1();
        }

        public static void TestAND()
        {
            var perceptron1 = new SimplePerceptron(2, 5, 1);
            var perceptron2 = new SimplePerceptron2(perceptron1);
            var perceptron3 = new PerceptronOld(perceptron1);

            Console.WriteLine("\nSimple Perceptron (AND)");
            TestAND(perceptron1);

            Console.WriteLine("\nPerceptron with bias (AND)");
            TestAND(perceptron2);

            Console.WriteLine("\nPerceptron old with bias and momentum (AND)");
            TestAND(perceptron3);
        }

        public static void TestNAND()
        {
            var perceptron1 = new SimplePerceptron(2, 1);
            var perceptron2 = new SimplePerceptron2(2, 3, 1);
            var perceptron3 = new PerceptronOld(2, 3, 1);

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
            var perceptron3 = new PerceptronOld(2, 3, 1);

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
            var perceptron3 = new PerceptronOld(2, 3, 1);

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
            var perceptron3 = new PerceptronOld(perceptron1);

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
            var perceptron3 = new PerceptronOld(perceptron1);

            Console.WriteLine("\nSimple Perceptron (Simple Numbers)");
            TestSimpleNumbers(perceptron1);

            Console.WriteLine("\nPerceptron with bias (Simple Numbers)");
            TestSimpleNumbers(perceptron2);

            Console.WriteLine("\nPerceptron with bias and momentum (Simple Numbers)");
            TestSimpleNumbers(perceptron3);
        }

        public static void TestAND(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateDataAND(), activationFunctionType);
        }

        public static void TestNAND(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateDataNAND(), activationFunctionType);
        }

        public static void TestOR(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateDataOR(), activationFunctionType);
        }

        public static void TestXOR(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TestLogic(perceptron, TrainData.GenerateDataXOR(), activationFunctionType);
        }

        public static void TestLogic(IPerceptronOld perceptron, TrainData trainData, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                trainData = CommonFunctions.CreateNewNormalizeTrainData(trainData, -1, 1);
            }

            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainData, LogicLearningRate, LogicTargetError, LogicMaxEpoch, LogicPrintError);
            Console.WriteLine(trainStats);

            TestReadyLogicModel(perceptron, activationFunctionType);
        }

        public static void TestLogic(IPerceptronBase perceptron, string title)
        {
            Console.WriteLine("\n###" + title);
            TestLogic(perceptron, TrainData.GenerateDataAND(), "AND");
            TestLogic(perceptron, TrainData.GenerateDataNAND(), "NAND");
            TestLogic(perceptron, TrainData.GenerateDataOR(), "OR");
            TestLogic(perceptron, TrainData.GenerateDataXOR(), "XOR");
        }

        public static void TestMultiplication(IPerceptronBase perceptron, TrainData trainData, string title)
        {
            Console.WriteLine("\n*" + title);
            var perceptronCopy = (IPerceptronBase)perceptron.Clone();
            var activationFunction = perceptronCopy.Topology.GetOutputActivationFunction();
            var perceptronTrainer = new PerceptronTrainer();
            var normalizedTrainData = trainData.GetNormalized(activationFunction.InputRange, activationFunction.OutputRange);
            var trainStats = perceptronTrainer.Train(perceptronCopy, normalizedTrainData, 0.5, 0.0005, 50000, LogicPrintError);
            Console.WriteLine(trainStats);
            IORanges ranges = CommonFunctions.GetMinMax(trainData);
            double[,] multiplicationTestData = new double[,]
            {
                { 2, 2 }, { 4, 4 }, { 2, 6 }, { 8, 8 }, { 9, 9 }, { 9, 3 },
                { 9, 0 }, { 0, 9 }, { 0, 3 }, { 0, 0 }, { 0, 1 }, { 5, 5 },
                { 1.5, 1.5 }, { 1.1, 1.1 }, { 8.9, 8.9 }, { 3.3, 3.3 }, { 0, 5.5 }
            };
            TrainData testData = TrainData.GenerateDataMultiplication(multiplicationTestData).GetNormalized(activationFunction.InputRange, activationFunction.OutputRange);
            TestReadyModel(perceptronCopy, testData, "*", ranges);
        }

        public static void TestSimpleNumbers(IPerceptronBase perceptron, TrainData trainData, string title)
        {
            Console.WriteLine("\n*" + title);
            var perceptronCopy = (IPerceptronBase)perceptron.Clone();
            var activationFunction = perceptronCopy.Topology.GetOutputActivationFunction();
            var perceptronTrainer = new PerceptronTrainer();
            var normalizedTrainData = trainData.GetNormalized(activationFunction.InputRange, activationFunction.OutputRange);
            var trainStats = perceptronTrainer.Train(perceptronCopy, normalizedTrainData, 0.5, 0.0005, 50000, LogicPrintError);
            Console.WriteLine(trainStats);
            IORanges ranges = CommonFunctions.GetMinMax(trainData);
            TestReadyModel(perceptronCopy, normalizedTrainData, "", ranges, TestReadyModelMode.LeftIsTargetOutput);
        }

        public static void TestLogic(IPerceptronBase perceptron, TrainData trainData, string title)
        {
            Console.WriteLine("\n*" + title);
            var perceptronCopy = (IPerceptronBase)perceptron.Clone();
            var activationFunction = perceptronCopy.Topology.GetOutputActivationFunction();
            var perceptronTrainer = new PerceptronTrainer();
            var normalizedTrainData = trainData.GetNormalized(activationFunction.InputRange, activationFunction.OutputRange);
            var trainStats = perceptronTrainer.Train(perceptronCopy, normalizedTrainData, LogicLearningRate, LogicTargetError, LogicMaxEpoch, LogicPrintError);
            Console.WriteLine(trainStats);
            IORanges ranges = CommonFunctions.GetMinMax(trainData);
            TestReadyModel(perceptronCopy, normalizedTrainData, " ", ranges);
        }

        public static void TestMultiplication(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            TrainData trainDataOrigin = TrainData.GenerateDataMultiplication();
            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);
            Interval sourceIntervalX = new Interval(xMin, xMax);
            Interval sourceIntervalY = new Interval(yMin, yMax);

            Interval normalizedIntervalX = activationFunctionType == ActivationFunctionType.Tanh ?
                new Interval(-1, 1) :
                new Interval(0, 1);
            Interval normalizedIntervalY = activationFunctionType == ActivationFunctionType.Tanh ?
                new Interval(-1, 1) :
                new Interval(0, 1);
            TrainData trainDataNormalized = activationFunctionType == ActivationFunctionType.Tanh ? 
                CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin, -1, 1) : 
                CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);

            Console.WriteLine("\nMultiplication training...");
            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainDataNormalized, 0.5, 0.0005, 50000, false);
            Console.WriteLine(trainStats);

            Console.WriteLine("\nMultiplication result");
            double[,] multiplicationTestData = new double[,]
            {
                { 2, 2 }, { 4, 4 }, { 2, 6 }, { 8, 8 }, { 9, 9 }, { 9, 3 },
                { 9, 0 }, { 0, 9 }, { 0, 3 }, { 0, 0 }, { 0, 1 }, { 5, 5 },
                { 1.5, 1.5 }, { 1.1, 1.1 }, { 8.9, 8.9 }, { 3.3, 3.3 }, { 0, 5.5 }
            };
            Interval[] intervalsX = new Interval[] { sourceIntervalX, normalizedIntervalX };
            Interval[] intervalsY = new Interval[] { sourceIntervalY, normalizedIntervalY };
            TestReadyModel2x(perceptron, multiplicationTestData, "*", intervalsX, intervalsY);
        }

        public static void TestSimpleNumbers(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid, double learningRate = 0.5, double targetError = 0.001)
        {
            TrainData trainDataOrigin = TrainData.GenerateDataSimpleNumbers();
            CommonFunctions.GetMinMax(trainDataOrigin.Inputs, out double xMin, out double xMax);
            CommonFunctions.GetMinMax(trainDataOrigin.Outputs, out double yMin, out double yMax);
            Interval sourceIntervalX = new Interval(xMin, xMax);
            Interval sourceIntervalY = new Interval(yMin, yMax);

            Interval normalizedIntervalX = activationFunctionType == ActivationFunctionType.Tanh ?
                new Interval(-1, 1) :
                new Interval(0, 1);
            Interval normalizedIntervalY = activationFunctionType == ActivationFunctionType.Tanh ?
                new Interval(-1, 1) :
                new Interval(0, 1);
            TrainData trainDataNormalized = activationFunctionType == ActivationFunctionType.Tanh ?
                CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin, -1, 1) :
                CommonFunctions.CreateNewNormalizeTrainData(trainDataOrigin);

            Console.WriteLine("\nSimpleNumbers training...");
            var perceptronTrainer = new PerceptronTrainer();
            TrainStats trainStats = perceptronTrainer.Train(perceptron, trainDataNormalized, learningRate, targetError, 50000, false);
            Console.WriteLine(trainStats);

            Console.WriteLine("\nSimpleNumbers result");
            double[] row = new double[trainDataOrigin.Inputs.GetLength(1)];
            for (int i = 0; i < trainDataNormalized.Inputs.GetLength(0); i++)
            {
                for (int c = 0; c < trainDataNormalized.Inputs.GetLongLength(1); c++)
                {
                    row[c] = trainDataNormalized.Inputs[i, c];
                }
                var output = perceptron.Forward(row);
                var denormalizedOutput = CommonFunctions.Scale(output, normalizedIntervalY, sourceIntervalY);
                Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("test {0}: {1:0.00}", i, y))));
            }
        }

        public static void TestReadyModel2x(IPerceptronOld perceptron, double[,] testData, string separate, Interval[] intervalsX, Interval[] intervalsY)
        {
            for (int i = 0; i < testData.GetLength(0); i++)
            {
                TestReadyModel2x(perceptron, testData[i, 0], testData[i, 1], separate, intervalsX, intervalsY);
            }
        }

        public static void TestReadyModel2x(IPerceptronOld perceptron, TrainData testData, string separate, Interval[] intervalsX, Interval[] intervalsY)
        {
            for (int i = 0; i < testData.Inputs.GetLength(0); i++)
            {
                TestReadyModel2x(perceptron, testData.Inputs[i, 0], testData.Inputs[i, 1], separate, intervalsX, intervalsY);
            }
        }

        public static void TestReadyModel2x(IPerceptronOld perceptron, double x1, double x2, string separate, Interval[] intervalsX, Interval[] intervalsY)
        {
            var output = perceptron.Forward(new double[] { CommonFunctions.Scale(x1, intervalsX[0], intervalsX[1]), CommonFunctions.Scale(x2, intervalsX[0], intervalsX[1]) });
            var denormalizedOutput = CommonFunctions.Scale(output, intervalsY[1], intervalsY[0]);
            Console.WriteLine(string.Join(" ", denormalizedOutput.Select(y => string.Format("{1} {0} {2} = {3}", separate, x1, x2, y))));
        }

        public static void TestReadyLogicModel(IPerceptronOld perceptron, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                var sample = new double[] { -1, -1 };
                var output = perceptron.Forward(sample);
                Console.WriteLine("0 0 = " + string.Join(" ", CommonFunctions.Scale(output, -1, 1, 0, 1).Select(x => Math.Round(x, 2))));

                sample = new double[] { -1, 1 };
                output = perceptron.Forward(sample);
                Console.WriteLine("0 1 = " + string.Join(" ", CommonFunctions.Scale(output, -1, 1, 0, 1).Select(x => Math.Round(x, 2))));

                sample = new double[] { 1, -1 };
                output = perceptron.Forward(sample);
                Console.WriteLine("1 0 = " + string.Join(" ", CommonFunctions.Scale(output, -1, 1, 0, 1).Select(x => Math.Round(x, 2))));

                sample = new double[] { 1, 1 };
                output = perceptron.Forward(sample);
                Console.WriteLine("1 1 = " + string.Join(" ", CommonFunctions.Scale(output, -1, 1, 0, 1).Select(x => Math.Round(x, 2))));
            }
            else
            {
                var sample = new double[] { 0, 0 };
                var output = perceptron.Forward(sample);
                Console.WriteLine("0 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

                sample = new double[] { 0, 1 };
                output = perceptron.Forward(sample);
                Console.WriteLine("0 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

                sample = new double[] { 1, 0 };
                output = perceptron.Forward(sample);
                Console.WriteLine("1 0 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));

                sample = new double[] { 1, 1 };
                output = perceptron.Forward(sample);
                Console.WriteLine("1 1 = " + string.Join(" ", output.Select(x => Math.Round(x, 2))));
            }
        }

        public static void TestReadyModel(IPerceptronBase perceptron, TrainData testData, string separateInput, IORanges ranges, TestReadyModelMode mode = TestReadyModelMode.LeftIsTargetInput)
        {
            var inputActivationFunction = perceptron.Topology.GetInputActivationFunction();
            var outputActivationFunction = perceptron.Topology.GetOutputActivationFunction();
            for (int i = 0; i < testData.Count; i++)
            {
                double[][] row = testData.GetRow(i);
                double[] output = perceptron.Forward(row[0]);
                double[] scaledRight = CommonFunctions.Scale(output, outputActivationFunction.OutputRange, ranges.OutputRange);
                if (mode == TestReadyModelMode.LeftIsTargetInput)
                {
                    double[] scaledLeft = CommonFunctions.Scale(row[0], inputActivationFunction.InputRange, ranges.InputRange);
                    Console.WriteLine(
                        "{0} = {1}",
                        string.Join(separateInput, scaledLeft),
                        string.Join(" ", scaledRight.Select(x => Math.Round(x, 2)))
                    );
                }
                else
                {
                    double[] scaledLeft = CommonFunctions.Scale(row[1], inputActivationFunction.OutputRange, ranges.OutputRange);
                    Console.WriteLine(
                        "{0} = {1}",
                        string.Join(separateInput, scaledLeft),
                        string.Join(" ", scaledRight.Select(x => Math.Round(x, 2)))
                    );
                }
            }
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

        public static void Comparison()
        {
            Console.WriteLine("\n\n#### Comparison Test");

            var perceptronTrainer = new PerceptronTrainer();

            var perceptronOld = new PerceptronOld(new int[] { 2, 5, 1 }, new Random(62784123));
            PerceptronTopology perceptronTopology = new PerceptronTopology(new int[] { 2, 5, 1 }, new ActivationFunction(ActivationFunctionType.Sigmoid));
            var perceptron = new Perceptron(perceptronOld);
            var perceptronBase = new PerceptronBase(perceptron);

            double[] errors = new double[3]; 

            TrainStats singleTrainStats = perceptronTrainer.Train(perceptronOld, TrainData.GenerateDataXOR(), 0.5, 1e-6, 500000, false);
            Console.WriteLine("{0}: {1}", "Perceptron Old", singleTrainStats);
            errors[0] = singleTrainStats.LastError;

            singleTrainStats = perceptronTrainer.Train(perceptronBase, TrainData.GenerateDataXOR(), 0.5, 1e-6, 500000, false);
            Console.WriteLine("{0}: {1}", "Perceptron Base", singleTrainStats);
            errors[1] = singleTrainStats.LastError;

            singleTrainStats = perceptronTrainer.Train(perceptron, TrainData.GenerateDataXOR(), 0.5, 1e-6, 500000, false);
            Console.WriteLine("{0}: {1}", "Perceptron", singleTrainStats);
            errors[2] = singleTrainStats.LastError;

            bool isValid = errors[0] == errors[1] && errors[0] == errors[2];

            Console.WriteLine(isValid ? "Is valid" : "IS NOT VALID!!!");
        }

        public enum TestReadyModelMode
        {
            LeftIsTargetInput,
            LeftIsTargetOutput
        }
    }
}
