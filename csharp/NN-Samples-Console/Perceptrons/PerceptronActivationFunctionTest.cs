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
            Console.WriteLine("\n########################################");
            Console.WriteLine("#### Perceptron Activation Functions ###");
            Console.WriteLine("########################################");

            PerceptronTypeTest.LogicLearningRate = 0.1;

            //TestLogic();
            //TestMultiplication();
            TestSimpleNumbers();
        }

        public static void TestLogic()
        {
            Perceptron perceptronAND1  = new Perceptron(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptronAND2  = new Perceptron(perceptronAND1, ActivationFunctionType.Tanh);
            Perceptron perceptronAND3  = new Perceptron(perceptronAND1, ActivationFunctionType.ReLU);
            Perceptron perceptronAND4  = new Perceptron(perceptronAND1, ActivationFunctionType.LReLU);
            Perceptron perceptronAND5  = new Perceptron(perceptronAND1, ActivationFunctionType.RandomLReLU);

            Perceptron perceptronNAND1 = new Perceptron(perceptronAND1, ActivationFunctionType.Sigmoid);
            Perceptron perceptronNAND2 = new Perceptron(perceptronAND1, ActivationFunctionType.Tanh);
            Perceptron perceptronNAND3 = new Perceptron(perceptronAND1, ActivationFunctionType.ReLU);
            Perceptron perceptronNAND4 = new Perceptron(perceptronAND1, ActivationFunctionType.LReLU);
            Perceptron perceptronNAND5 = new Perceptron(perceptronAND1, ActivationFunctionType.RandomLReLU);

            Perceptron perceptronOR1   = new Perceptron(perceptronAND1, ActivationFunctionType.Sigmoid);
            Perceptron perceptronOR2   = new Perceptron(perceptronAND1, ActivationFunctionType.Tanh);
            Perceptron perceptronOR3   = new Perceptron(perceptronAND1, ActivationFunctionType.ReLU);
            Perceptron perceptronOR4   = new Perceptron(perceptronAND1, ActivationFunctionType.LReLU);
            Perceptron perceptronOR5   = new Perceptron(perceptronAND1, ActivationFunctionType.RandomLReLU);

            Perceptron perceptronXOR1  = new Perceptron(perceptronAND1, ActivationFunctionType.Sigmoid);
            Perceptron perceptronXOR2  = new Perceptron(perceptronAND1, ActivationFunctionType.Tanh);
            Perceptron perceptronXOR3  = new Perceptron(perceptronAND1, ActivationFunctionType.ReLU);
            Perceptron perceptronXOR4  = new Perceptron(perceptronAND1, ActivationFunctionType.LReLU);
            Perceptron perceptronXOR5  = new Perceptron(perceptronAND1, ActivationFunctionType.RandomLReLU);


            Console.WriteLine("\n\n#### Test AND");
            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestAND(perceptronAND1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestAND(perceptronAND2, ActivationFunctionType.Tanh);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestAND(perceptronAND3);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestAND(perceptronAND4);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestAND(perceptronAND5);


            Console.WriteLine("\n\n#### Test NAND");
            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestNAND(perceptronNAND1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestNAND(perceptronNAND2, ActivationFunctionType.Tanh);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestNAND(perceptronNAND3);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestNAND(perceptronNAND4);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestNAND(perceptronNAND5);


            Console.WriteLine("\n\n#### Test OR");
            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestOR(perceptronOR1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestOR(perceptronOR2, ActivationFunctionType.Tanh);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestOR(perceptronOR3);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestOR(perceptronOR4);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestOR(perceptronOR5);


            Console.WriteLine("\n\n#### Test XOR");
            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestXOR(perceptronXOR1);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestXOR(perceptronXOR2, ActivationFunctionType.Tanh);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestXOR(perceptronXOR3);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestXOR(perceptronXOR4);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestXOR(perceptronXOR5);
        }
        
        public static void TestMultiplication()
        {
            Console.WriteLine("\n\n#### Test Multiplication");

            Perceptron perceptron1 = new Perceptron(new int[] { 2, 24, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.Tanh);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);
            Perceptron perceptron4 = new Perceptron(perceptron1, ActivationFunctionType.LReLU);
            Perceptron perceptron5 = new Perceptron(perceptron1, ActivationFunctionType.RandomLReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestMultiplication(perceptron1, ActivationFunctionType.Sigmoid);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestMultiplication(perceptron2, ActivationFunctionType.Tanh);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestMultiplication(perceptron3, ActivationFunctionType.ReLU);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestMultiplication(perceptron4, ActivationFunctionType.LReLU);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestMultiplication(perceptron5, ActivationFunctionType.RandomLReLU);
        }

        public static void TestSimpleNumbers(double learningRate = 0.2, double targetError = 1e-7)
        {
            Console.WriteLine("\n\n#### Test SimpleNumbers");

            Perceptron perceptron1 = new Perceptron(new int[] { 35, 32, 16, 1 }, ActivationFunctionType.Sigmoid);
            Perceptron perceptron2 = new Perceptron(perceptron1, ActivationFunctionType.Tanh);
            Perceptron perceptron3 = new Perceptron(perceptron1, ActivationFunctionType.ReLU);
            Perceptron perceptron4 = new Perceptron(perceptron1, ActivationFunctionType.LReLU);
            Perceptron perceptron5 = new Perceptron(perceptron1, ActivationFunctionType.RandomLReLU);

            Console.WriteLine("\nSigmoid");
            PerceptronTypeTest.TestSimpleNumbers(perceptron1, ActivationFunctionType.Sigmoid, learningRate, targetError);

            Console.WriteLine("\nHyperbolic Tangent");
            PerceptronTypeTest.TestSimpleNumbers(perceptron2, ActivationFunctionType.Tanh, learningRate, targetError);

            Console.WriteLine("\nReLU");
            PerceptronTypeTest.TestSimpleNumbers(perceptron3, ActivationFunctionType.ReLU, learningRate, targetError);

            Console.WriteLine("\nLReLU");
            PerceptronTypeTest.TestSimpleNumbers(perceptron4, ActivationFunctionType.LReLU, learningRate, targetError);

            Console.WriteLine("\nRandom LReLU");
            PerceptronTypeTest.TestSimpleNumbers(perceptron5, ActivationFunctionType.RandomLReLU, learningRate, targetError);
        }
    }
}
