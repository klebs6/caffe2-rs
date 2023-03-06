crate::ix!();

/**
  | Computes a histogram for values in the
  | given list of tensors.
  | 
  | For logging activation histograms
  | for post-hoc analyses, consider using
  | the HistogramObserver observer.
  | 
  | For iteratively computing a histogram
  | for all input tensors encountered through
  | history, consider using the AccumulateHistogram
  | operator.
  |
  */
pub struct HistogramOp<Context> {
    storage: OperatorStorage,
    context: Context,
    bin_edges: Vec<f32>,
}

pub enum HistogramOpOutputs {
    Histogram,
}

register_cpu_operator!{Histogram, HistogramOp<CPUContext>}

num_inputs!{Histogram, (1,INT_MAX)}

num_outputs!{Histogram, 1}

inputs!{Histogram, 
    0 => ("X1, X2, ...", "*(type: Tensor`<float>`)* List of input tensors.")
}

outputs!{Histogram, 
    0 => ("histogram", 
        "1D tensor of length k, wherein the i-th element expresses the 
        count of tensor values that fall within range [bin_edges[i], bin_edges[i + 1])")
}

args!{Histogram, 
    0 => ("bin_edges", 
        "length-(k + 1) sequence of float values wherein the i-th element 
        represents the inclusive left boundary of the i-th bin for i in [0, k - 1] and 
        the exclusive right boundary of the (i-1)-th bin for i in [1, k].")
}

should_not_do_gradient!{Histogram}
