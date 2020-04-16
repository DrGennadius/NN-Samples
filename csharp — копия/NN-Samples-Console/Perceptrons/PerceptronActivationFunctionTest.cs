using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives;
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

            TestLogic();
            TestMultiplication();
            TestSimpleNumbers();
        }

        public static void TestLogic()
        {
            PerceptronOld perceptronAND1  = new PerceptronOld(new int[] { 2, 5, 1 }, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronAND2  = new PerceptronOld(perceptronAND1, ActivationFunctionType.Tanh);
            PerceptronOld perceptronAND3  = new PerceptronOld(perceptronAND1, ActivationFunctionType.ReLU);
            PerceptronOld perceptronAND4  = new PerceptronOld(perceptronAND1, ActivationFunctionType.LReLU);
            PerceptronOld perceptronAND5  = new PerceptronOld(perceptronAND1, ActivationFunctionType.RandomLReLU);

            PerceptronOld perceptronNAND1 = new PerceptronOld(perceptronAND1, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronNAND2 = new PerceptronOld(perceptronAND1, ActivationFunctionType.Tanh);
            PerceptronOld perceptronNAND3 = new PerceptronOld(perceptronAND1, ActivationFunctionType.ReLU);
            PerceptronOld perceptronNAND4 = new PerceptronOld(perceptronAND1, ActivationFunctionType.LReLU);
            PerceptronOld perceptronNAND5 = new PerceptronOld(perceptronAND1, ActivationFunctionType.RandomLReLU);

            PerceptronOld perceptronOR1   = new PerceptronOld(perceptronAND1, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronOR2   = new PerceptronOld(perceptronAND1, ActivationFunctionType.Tanh);
            PerceptronOld perceptronOR3   = new PerceptronOld(perceptronAND1, ActivationFunctionType.ReLU);
            PerceptronOld perceptronOR4   = new PerceptronOld(perceptronAND1, ActivationFunctionType.LReLU);
            PerceptronOld perceptronOR5   = new PerceptronOld(perceptronAND1, ActivationFunctionType.RandomLReLU);

            PerceptronOld perceptronXOR1  = new PerceptronOld(perceptronAND1, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptronXOR2  = new PerceptronOld(perceptronAND1, ActivationFunctionType.Tanh);
            PerceptronOld perceptronXOR3  = new PerceptronOld(perceptronAND1, ActivationFunctionType.ReLU);
            PerceptronOld perceptronXOR4  = new PerceptronOld(perceptronAND1, ActivationFunctionType.LReLU);
            PerceptronOld perceptronXOR5  = new PerceptronOld(perceptronAND1, ActivationFunctionType.RandomLReLU);


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

            PerceptronOld perceptron1 = new PerceptronOld(new int[] { 2, 24, 1 }, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptron2 = new PerceptronOld(perceptron1, ActivationFunctionType.Tanh);
            PerceptronOld perceptron3 = new PerceptronOld(perceptron1, ActivationFunctionType.ReLU);
            PerceptronOld perceptron4 = new PerceptronOld(perceptron1, ActivationFunctionType.LReLU);
            PerceptronOld perceptron5 = new PerceptronOld(perceptron1, ActivationFunctionType.RandomLReLU);

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

            PerceptronOld perceptron1 = new PerceptronOld(new int[] { 35, 32, 16, 1 }, ActivationFunctionType.Sigmoid);
            PerceptronOld perceptron2 = new PerceptronOld(perceptron1, ActivationFunctionType.Tanh);
            PerceptronOld perceptron3 = new PerceptronOld(perceptron1, ActivationFunctionType.ReLU);
            PerceptronOld perceptron4 = new PerceptronOld(perceptron1, ActivationFunctionType.LReLU);
            PerceptronOld perceptron5 = new PerceptronOld(perceptron1, ActivationFunctionType.RandomLReLU);

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
