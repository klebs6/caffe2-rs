crate::ix!();

/**
  | Computes a histogram for values in the
  | given list of tensors.
  | 
  | For logging activation histograms
  | for post-hoc analyses, consider using
  | the
  | 
  | HistogramObserver observer.
  | 
  | For iteratively computing a histogram
  | for all input tensors encountered through
  | history, consider using the AccumulateHistogram
  | operator.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SelfBinningHistogramOp<Context> {

    context:         Context,

    num_bins:        i32,
    num_edges:       i32,
    bin_spacing:     String,
    logspace_start:  f32,

    /**
      | automatically apply abs() on the input
      | values
      |
      */
    abs:             bool,
}

output_tags!{
    SelfBinningHistogramOp {
        HistogramValues,
        HistogramCounts
    }
}

register_cpu_operator!{SelfBinningHistogram, SelfBinningHistogramOp<CPUContext>}

num_inputs!{SelfBinningHistogram, (1,INT_MAX)}

num_outputs!{SelfBinningHistogram, 2}

inputs!{SelfBinningHistogram, 
    0 => ("X1, X2, ...",      "*(type: Tensor`<float>`)* List of input tensors.")
}

outputs!{SelfBinningHistogram, 
    0 => ("histogram_values", "1D tensor of edges of the bins, of dimension [num_bins+1]. The range appears as: [first, ..., last), wherein the i-th element expresses the start of a bin and i+1-th value represents the exclusive end of that bin."),
    1 => ("histogram_counts", "1D tensor of counts of each bin, of dimension [num_bins+1]. It is guaranteed to end with a 0 since the last edge is exclusive.")
}

args!{SelfBinningHistogram, 
    0 => ("num_bins",         "Number of bins to use for the histogram. Must be >= 1."),
    1 => ("bin_spacing",      "A string indicating 'linear' or 'logarithmic' spacing for the bins."),
    2 => ("logspace_start",   "A float that's used as the starting point for logarithmic spacing. Since logarithmic spacing cannot contain <=0 values this value will be used to represent all such values."),
    3 => ("abs",              "Apply abs() on every input value.")
}

should_not_do_gradient!{SelfBinningHistogram}
