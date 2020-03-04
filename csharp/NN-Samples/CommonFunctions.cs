using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples
{
    public class CommonFunctions
    {
        public static double GeneralError(double[,] realOutputs, double[,] goalOutputs)
        {
            double err = 0;
            for (int i = 0; i < realOutputs.GetLength(0); i++)
            {
                double[] realOutputsRow = new double[realOutputs.GetLength(1)];
                double[] goalOutputsRow = new double[goalOutputs.GetLength(1)];
                for (var c = 0; c < realOutputs.GetLength(1); c++)
                {
                    realOutputsRow[c] = realOutputs[i, c];
                }
                for (var c = 0; c < goalOutputs.GetLength(1); c++)
                {
                    goalOutputsRow[c] = goalOutputs[i, c];
                }
                err += IndividualError(realOutputsRow, goalOutputsRow);
            }
            return err;
        }

        public static double IndividualError(double[] realOutputs, double[] goalOutputs)
        {
            double err = 0;
            for (int i = 0; i < realOutputs.Length; i++)
            {
                err += Math.Pow(realOutputs[i] - goalOutputs[i], 2);
            }
            return err;
        }

        public static void GetMinMax(double[] values, out double min, out double max)
        {
            double cmin = values[0];
            double cmax = values[0];
            foreach (double value in values)
            {
                if (value > cmax)
                {
                    cmax = value;
                }
                if (value < cmin)
                {
                    cmin = value;
                }
            }
            min = cmin;
            max = cmax;
        }

        public static void GetMinMax(double[,] values, out double min, out double max)
        {
            GetMinMax(Flatten(values), out min, out max);
        }

        public static double Denormalize(double val, double min, double max)
        {
            return val * (max - min) + min;
        }

        public static double[] Denormalize(double[] values, double min, double max)
        {
            double[] newValues = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                newValues[i] = Denormalize(values[i], min, max);
            }
            return newValues;
        }

        public static double Normalize(double val, double min, double max)
        {
            return (val - min) / (max - min);
        }

        public static double[] Normalize(double[] values)
        {
            double[] normalizedValues = new double[values.Length];
            GetMinMax(values, out double min, out double max);
            for (int i = 0; i < values.Length; i++)
            {
                normalizedValues[i] = Normalize(values[i], min, max);
            }
            return normalizedValues;
        }

        public static double[,] Normalize(double[,] values)
        {
            int size1 = values.GetLength(0);
            int size2 = values.GetLength(1);
            double[] flattenNormalizedValues = Normalize(Flatten(values));
            double[,] normalizedValues = new double[size1, size2];
            for (int i = 0; i < size1; i++)
            {
                for (int k = 0; k < size2; k++)
                {
                    normalizedValues[i, k] = flattenNormalizedValues[i * size2 + k];
                }
            }
            return normalizedValues;
        }

        public static TrainData NormalizeTrainData(TrainData trainData)
        {
            trainData.Inputs = Normalize(trainData.Inputs);
            trainData.Outputs = Normalize(trainData.Outputs);
            return trainData;
        }

        public static TrainData CreateNewNormalizeTrainData(TrainData trainData)
        {
            TrainData newTrainData = new TrainData();
            newTrainData.Inputs = Normalize(trainData.Inputs);
            newTrainData.Outputs = Normalize(trainData.Outputs);
            return newTrainData;
        }

        public static double[] Flatten(double[,] values)
        {
            int size1 = values.GetLength(0);
            int size2 = values.GetLength(1);
            double[] result = new double[size1 * size2];
            int write = 0;
            for (int i = 0; i < size1; i++)
            {
                for (int z = 0; z < size2; z++)
                {
                    result[write++] = values[i, z];
                }
            }
            return result;
        }
    }
}
