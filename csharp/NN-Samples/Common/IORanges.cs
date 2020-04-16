using System;
using System.Collections.Generic;
using System.Text;

namespace NN_Samples.Common
{
    public struct IORanges
    {
        public IORanges(Range<double> inputRange, Range<double> outputRange)
        {
            InputRange = inputRange;
            OutputRange = outputRange;
        }

        public Range<double> InputRange { get; set; }
        public Range<double> OutputRange { get; set; }
    }
}
