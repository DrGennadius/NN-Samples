using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    public class ActivationFunctions
    {
        /// <summary>
        /// Sigmoid. Returns a value between 0 and 1.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivated(double x)
        {
            return x * (1 - x);
        }

        /// <summary>
        /// Hyperbolic Tangent. Returns a value between -1 and +1.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double HyperbolicTangent(double x)
        {
            return Math.Tanh(x);
        }

        public static double HyperbolicTangentDerivated(double x)
        {
            return 1 - x * x;
        }

        public static double ReLU(double x)
        {
            return x > 0 ? x : 0;
        }

        public static double ReLUDerivated(double x)
        {
            return x > 0 ? 1 : 0;
        }
    }

    public struct ActivationFunction
    {
        public ActivationFunction(ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            if (activationFunctionType == ActivationFunctionType.Sigmoid)
            {
                Calculate = ActivationFunctions.Sigmoid;
                CalculateDerivative = ActivationFunctions.SigmoidDerivated;
            }
            else if (activationFunctionType == ActivationFunctionType.HyperbolicTangent)
            {
                Calculate = ActivationFunctions.HyperbolicTangent;
                CalculateDerivative = ActivationFunctions.HyperbolicTangent;
            }
            else
            {
                Calculate = ActivationFunctions.ReLU;
                CalculateDerivative = ActivationFunctions.ReLUDerivated;
            }
        }

        public Func<double, double> Calculate;

        public Func<double, double> CalculateDerivative;
    }

    public enum ActivationFunctionType
    {
        Sigmoid,
        HyperbolicTangent,
        ReLU
    }
}
