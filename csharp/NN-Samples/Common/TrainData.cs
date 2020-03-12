using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    /// <summary>
    /// Data for training.
    /// </summary>
    public class TrainData
    {
        public double[,] Inputs;

        public double[,] Outputs;

        /// <summary>
        /// Generate data "XOR".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataXOR()
        {
            TrainData trainData = new TrainData();
            trainData.Inputs = new double[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            };
            trainData.Outputs = new double[,]
            {
                { 0 },
                { 1 },
                { 1 },
                { 0 }
            };
            return trainData;
        }

        /// <summary>
        /// Generate data "OR".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataOR()
        {
            TrainData trainData = new TrainData();
            trainData.Inputs = new double[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            };
            trainData.Outputs = new double[,]
            {
                { 0 },
                { 1 },
                { 1 },
                { 1 }
            };
            return trainData;
        }

        /// <summary>
        /// Generate data "AND".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataAND()
        {
            TrainData trainData = new TrainData();
            trainData.Inputs = new double[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            };
            trainData.Outputs = new double[,]
            {
                { 0 },
                { 0 },
                { 0 },
                { 1 }
            };
            return trainData;
        }

        /// <summary>
        /// Generate data "NAND".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataNAND()
        {
            TrainData trainData = new TrainData();
            trainData.Inputs = new double[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            };
            trainData.Outputs = new double[,]
            {
                { 1 },
                { 1 },
                { 1 },
                { 0 }
            };
            return trainData;
        }

        /// <summary>
        /// Generate data "Multiplication".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataMultiplication()
        {
            TrainData trainData = new TrainData();
            int min = 0;
            int max = 9;
            int p = 0;
            for (int i = min; i <= max; i++)
            {
                p++;
            }
            p *= p;
            trainData.Inputs = new double[p, 2];
            trainData.Outputs = new double[p, 1];
            int count = 0;
            for (int i = min; i <= max; i++)
            {
                for (int k = min; k <= max; k++)
                {
                    trainData.Inputs[count, 0] = i;
                    trainData.Inputs[count, 1] = k;
                    trainData.Outputs[count, 0] = i * k;
                    count++;
                }
            }
            return trainData;
        }

        /// <summary>
        /// Generate data "Simple Numbers".
        /// </summary>
        /// <returns></returns>
        public static TrainData GenerateDataSimpleNumbers()
        {
            TrainData trainData = new TrainData();
            trainData.Inputs = new double[,]
            {
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 1, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 0, 0
                },
                {
                    0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0,
                    0, 1, 0, 1, 0,
                    0, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0
                }
            };
            trainData.Outputs = new double[,]
            {
                { 0 },
                { 1 },
                { 2 },
                { 3 },
                { 4 },
                { 5 },
                { 6 },
                { 7 },
                { 8 },
                { 9 }
            };
            return trainData;
        }
    }
}
