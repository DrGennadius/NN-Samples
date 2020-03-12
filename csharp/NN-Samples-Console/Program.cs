using NN_Samples;
using NN_Samples.Common;
using NN_Samples.Perceptrons;
using NN_Samples.Perceptrons.Alternatives.Samples;
using NN_Samples_Console.Perceptrons;
using System;
using System.Linq;

namespace NN_Samples_Console
{
    class Program
    {
        static void Main(string[] args)
        {
            PerceptronTypeTest.Test();
            //PerceptronActivationFunctionTest.Test();

            Console.ReadKey();
        }
    }
}
