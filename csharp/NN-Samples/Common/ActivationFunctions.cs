using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    public class ActivationFunctions
    {
        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public static double SigmoidDerivated(double input)
        {
            double y = Sigmoid(input);
            return y * (1 - y);
        }
    }
}
