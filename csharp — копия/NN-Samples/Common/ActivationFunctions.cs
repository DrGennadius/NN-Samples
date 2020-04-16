using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    /// <summary>
    /// Activation functions.
    /// </summary>
    public class ActivationFunctions
    {
        /// <summary>
        /// Get random number from minimum to maximum
        /// </summary>
        /// <param name="minimum"></param>
        /// <param name="maximum"></param>
        /// <returns></returns>
        public static double GetRandomNumber(double minimum, double maximum)
        {
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

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

        /// <summary>
        /// Rectified Linear Unit.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double ReLU(double x)
        {
            return x > 0 ? x : 0;
        }

        public static double ReLUDerivated(double x)
        {
            return x > 0 ? 1 : 0.01;
        }

        /// <summary>
        /// Derivative of Leaky ReLU with 0.01.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double LReLUDerivated(double x)
        {
            return x > 0 ? 1 : 0.01;
        }

        /// <summary>
        /// Derivative of Leaky ReLU with random 0.001 to 0.05.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double RandomLReLUDerivated(double x)
        {
            return x > 0 ? 1 : GetRandomNumber(0.001, 0.05);
        }
    }

    /// <summary>
    /// Activation function. Contains direct and derivative methods (delegates).
    /// </summary>
    public struct ActivationFunction
    {
        public ActivationFunction(Func<double, double> func, Func<double, double> derivativeFunc, Range<double> inputRange, Range<double> outputRange)
        {
            Type = ActivationFunctionType.Custom;
            Calculate = func;
            CalculateDerivative = derivativeFunc;
            InputRange = inputRange;
            OutputRange = outputRange;
        }

        public ActivationFunction(ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            Type = activationFunctionType;
            if (activationFunctionType == ActivationFunctionType.Sigmoid)
            {
                Calculate = ActivationFunctions.Sigmoid;
                CalculateDerivative = ActivationFunctions.SigmoidDerivated;
                InputRange = new Range<double>(0, 1);
                OutputRange = new Range<double>(0, 1);
            }
            else if (activationFunctionType == ActivationFunctionType.Tanh)
            {
                Calculate = ActivationFunctions.HyperbolicTangent;
                CalculateDerivative = ActivationFunctions.HyperbolicTangentDerivated;
                InputRange = new Range<double>(-1, 1);
                OutputRange = new Range<double>(-1, 1);
            }
            else if (activationFunctionType == ActivationFunctionType.ReLU)
            {
                Calculate = ActivationFunctions.ReLU;
                CalculateDerivative = ActivationFunctions.ReLUDerivated;
                InputRange = new Range<double>(0, 1);
                OutputRange = new Range<double>(0, 1);
            }
            else if (activationFunctionType == ActivationFunctionType.LReLU)
            {
                Calculate = ActivationFunctions.ReLU;
                CalculateDerivative = ActivationFunctions.LReLUDerivated;
                InputRange = new Range<double>(0, 1);
                OutputRange = new Range<double>(0, 1);
            }
            else
            {
                Calculate = ActivationFunctions.ReLU;
                CalculateDerivative = ActivationFunctions.RandomLReLUDerivated;
                InputRange = new Range<double>(0, 1);
                OutputRange = new Range<double>(0, 1);
            }
        }

        public Func<double, double> Calculate { get; private set; }

        public Func<double, double> CalculateDerivative { get; private set; }

        public ActivationFunctionType Type { get; private set; }

        /// <summary>
        /// Recomended range for input data.
        /// </summary>
        public Range<double> InputRange { get; private set; }

        /// <summary>
        /// Recomended range for output data.
        /// </summary>
        public Range<double> OutputRange { get; private set; }
    }

    /// <summary>
    /// Type of activation function.
    /// </summary>
    public enum ActivationFunctionType
    {
        /// <summary>
        /// Sigmoid.
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Hyperbolic Tangent.
        /// </summary>
        Tanh,

        /// <summary>
        /// Rectified Linear Unit.
        /// </summary>
        ReLU,
        /// <summary>
        /// Leaky Rectified Linear Unit.
        /// </summary>
        LReLU,

        /// <summary>
        /// Leaky Rectified Linear Unit with random.
        /// </summary>
        RandomLReLU,

        /// <summary>
        /// Custom function.
        /// </summary>
        Custom
    }
}
