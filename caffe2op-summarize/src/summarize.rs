crate::ix!();

pub const kSummaryzeOpExtension: &'static str = ".summary";

/**
  | Summarize computes four statistics
  | of the input tensor (Tensor)- min, max,
  | mean and standard deviation.
  | 
  | The output will be written to a 1-D tensor
  | of size 4 if an output tensor is provided.
  | 
  | Else, if the argument 'to_file' is greater
  | than 0, the values are written to a log
  | file in the root folder.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SummarizeOp<T,Context> {

    storage:   OperatorStorage,
    context:   Context,

    to_file:   bool,
    log_file:  Box<std::fs::File>,

    /**
      | Input: X;
      |
      | output: if set, a summarized Tensor of
      | shape 4, with the values being min, max,
      | mean and std respectively.
      */
    phantom:   PhantomData<T>,
}

register_cpu_operator!{Summarize, SummarizeOp<float, CPUContext>}

num_inputs!{Summarize, 1}

num_outputs!{Summarize, (0,1)}

inputs!{Summarize, 
    0 => ("data", "The input data as Tensor.")
}

outputs!{Summarize, 
    0 => ("output", "1-D tensor (Tensor) of size 4 containing min, max, mean and standard deviation")
}

args!{Summarize, 
    0 => ("to_file", "(int, default 0) flag to indicate if the summarized statistics have to be written to a log file.")
}

should_not_do_gradient!{Summarize}
