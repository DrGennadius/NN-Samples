using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    public struct TrainStats
    {
        public double LastError;

        public int NumberOfEpoch;

        public override string ToString()
        {
            return string.Format("Last Error = {0}; Number of epoch = {1}", LastError, NumberOfEpoch);
        }
    }
}
