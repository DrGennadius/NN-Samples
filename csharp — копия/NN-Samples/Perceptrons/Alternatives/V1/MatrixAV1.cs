/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Matrix. Class based on programforyou.ru sample
    /// </summary>
    public class MatrixAV1
    {
        private double[][] Values;

        public int RowSize;
        public int ColumnSize;

        /// <summary>
        /// Creation of a matrix of a given size and filling with random numbers from the interval (-0.5, 0.5).
        /// </summary>
        /// <param name="rowSize"></param>
        /// <param name="columnSize"></param>
        /// <param name="random"></param>
        public MatrixAV1(int rowSize, int columnSize, Random random)
        {
            RowSize = rowSize;
            ColumnSize = columnSize;

            Values = new double[rowSize][];

            for (int i = 0; i < rowSize; i++)
            {
                Values[i] = new double[columnSize];

                for (int j = 0; j < columnSize; j++)
                {
                    Values[i][j] = random.NextDouble() - 0.5;
                }
            }
        }

        /// <summary>
        /// Creation of a matrix by data.
        /// </summary>
        /// <param name="data"></param>
        public MatrixAV1(double[][] data)
        {
            Values = data;

            RowSize = data.Length;
            ColumnSize = data[0].Length;
        }

        public double this[int i, int j]
        {
            get { return Values[i][j]; }
            set { Values[i][j] = value; }
        }
    }
}
