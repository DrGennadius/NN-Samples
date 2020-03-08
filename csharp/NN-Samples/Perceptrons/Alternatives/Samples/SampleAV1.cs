/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using NN_Samples.Perceptrons.Alternatives.V1;
using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Alternatives.Samples
{
    /// <summary>
    /// The sample based on programforyou.ru sample.
    /// </summary>
    public class SampleAV1
    {
        /// <summary>
        /// XOR
        /// </summary>
        /// <returns></returns>
        public static string RunXOR()
        {
            StringBuilder result = new StringBuilder();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 ^ 0 = 0
                new VectorAV1(1.0), // 0 ^ 1 = 1
                new VectorAV1(1.0), // 1 ^ 0 = 1
                new VectorAV1(0.0) // 1 ^ 1 = 0
            };

            PerceptronAV1 perceptron = new PerceptronAV1(new int[] { 2, 3, 1 }); // создаём сеть с двумя входами, тремя нейронами в скрытом слое и одним выходом

            perceptron.Train(X, Y, 0.5, 1e-7, 100000); // запускаем обучение сети 

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// AND
        /// </summary>
        /// <returns></returns>
        public static string RunAND()
        {
            StringBuilder result = new StringBuilder();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 and 0 = 0
                new VectorAV1(0.0), // 0 and 1 = 0
                new VectorAV1(0.0), // 1 and 0 = 0
                new VectorAV1(1.0) // 1 and 1 = 1
            };

            PerceptronAV1 perceptron = new PerceptronAV1(new int[] { 2, 3, 1 }); // создаём сеть с двумя входами, тремя нейронами в скрытом слое и одним выходом

            perceptron.Train(X, Y, 0.5, 1e-7, 100000); // запускаем обучение сети 

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }

        /// <summary>
        /// OR
        /// </summary>
        /// <returns></returns>
        public static string RunOR()
        {
            StringBuilder result = new StringBuilder();

            // array of input training vectors
            VectorAV1[] X = {
                new VectorAV1(0, 0),
                new VectorAV1(0, 1),
                new VectorAV1(1, 0),
                new VectorAV1(1, 1)
            };

            // array of output training vectors
            VectorAV1[] Y = {
                new VectorAV1(0.0), // 0 and 0 = 0
                new VectorAV1(1.0), // 0 and 1 = 1
                new VectorAV1(1.0), // 1 and 0 = 1
                new VectorAV1(1.0) // 1 and 1 = 1
            };

            PerceptronAV1 perceptron = new PerceptronAV1(new int[] { 2, 3, 1 }); // создаём сеть с двумя входами, тремя нейронами в скрытом слое и одним выходом

            perceptron.Train(X, Y, 0.5, 1e-7, 100000); // запускаем обучение сети 

            for (int i = 0; i < 4; i++)
            {
                VectorAV1 output = perceptron.Forward(X[i]);
                result.Append(string.Format("{0}X: {1} {2}, Y: {3}, output: {4}", i == 0 ? "" : "\n", X[i][0], X[i][1], Y[i][0], output[0]));
            }

            return result.ToString();
        }
    }
}
