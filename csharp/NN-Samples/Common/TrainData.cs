using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    public class TrainData
    {
        public double[,] Inputs;

        public double[,] Outputs;

        public static TrainData GenerateXORDate()
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

        public static TrainData GenerateORDate()
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

        public static TrainData GenerateANDDate()
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

        public static TrainData GenerateMultiplicationDate()
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

        public static TrainData GenerateSimpleNumbersDate()
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
