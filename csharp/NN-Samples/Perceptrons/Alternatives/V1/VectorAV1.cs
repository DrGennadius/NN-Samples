/****************************************************************************************
 * Based on https://programforyou.ru/poleznoe/pishem-neuroset-pryamogo-rasprostraneniya *
 ****************************************************************************************/

using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Perceptrons.Alternatives.V1
{
    /// <summary>
    /// Vector. Class based on programforyou.ru sample
    /// </summary>
    public class VectorAV1
    {
        private double[] Values;

        public int Length;

        /// <summary>
        /// Constructor with length.
        /// </summary>
        /// <param name="length"></param>
        public VectorAV1(int length)
        {
            Length = length;
            Values = new double[length];
        }

        /// <summary>
        /// Constructor with array of value.
        /// </summary>
        /// <param name="values"></param>
        public VectorAV1(params double[] values)
        {
            Length = values.Length;
            Values = new double[Length];

            for (int i = 0; i < Length; i++)
            {
                Values[i] = values[i];
            }
        }
        
        public double this[int index]
        {
            get { return Values[index]; }
            set { Values[index] = value; }
        }
    }
}
