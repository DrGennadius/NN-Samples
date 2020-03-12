using NN_Samples.Common;
using NN_Samples.Perceptrons;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples_Console.Perceptrons
{
    public class PerceptronActivationFunctionTest
    {
        public static void Test()
        {
            TestAND();
            TestNAND();
            TestOR();
            TestXOR();
            TestMultiplication();
            TestSimpleNumbers();
        }

        public static void TestAND()
        {
            Console.WriteLine("\nTest AND");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestAND(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestAND(perceptron2, ActivationFunctionType.HyperbolicTangent);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestAND(perceptron3);
        }

        public static void TestNAND()
        {
            Console.WriteLine("\nTest NAND");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestNAND(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestNAND(perceptron2);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestNAND(perceptron3);
        }

        public static void TestOR()
        {
            Console.WriteLine("\nTest OR");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestOR(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestOR(perceptron2);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestOR(perceptron3);
        }

        public static void TestXOR()
        {
            Console.WriteLine("\nTest XOR");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestXOR(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestXOR(perceptron2);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestXOR(perceptron3);
        }

        public static void TestMultiplication()
        {
            Console.WriteLine("\nTest Multiplication");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 64, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestMultiplication(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestMultiplication(perceptron2);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestMultiplication(perceptron3);
        }

        public static void TestSimpleNumbers()
        {
            Console.WriteLine("\nTest XOR");

            Perceptron perceptron1 = new Perceptron(new int[] { 35, 128, 32, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.HyperbolicTangent);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestSimpleNumbers(perceptron1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestSimpleNumbers(perceptron2);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestSimpleNumbers(perceptron3);
        }
    }
}
