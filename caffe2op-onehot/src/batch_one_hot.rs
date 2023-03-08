crate::ix!();

/**
  | Input is a matrix tensor. Its first dimension
  | is the batch size. Expand each column
  | of it using one hot encoding. The `lengths`
  | specifies the size of each column after
  | encoding, and the `values` is the dictionary
  | value of one-hot encoding for each column.
  | For example
  | 
  | If data = [[2, 3], [4, 1], [2, 5]], lengths
  | = [2, 3], and values = [2, 4, 1, 3, 5], then
  | 
  | output = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0],
  | [1, 0, 0, 0, 1]]
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchOneHotOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /**
      | allows for fast random access to a given
      | dict and is re-used across runs
      |
      */
    vals_offsets: Vec<i64>,
}

num_inputs!{BatchOneHot, 3}

num_outputs!{BatchOneHot, 1}

inputs!{BatchOneHot, 
    0 => ("data", "input tensor matrix"),
    1 => ("lengths", "the size is the same as the width of the `data`"),
    2 => ("values", "one hot encoding dictionary values")
}

outputs!{BatchOneHot, 
    0 => ("output", "output matrix that expands each input column with one hot encoding")
}

cost_inference_function!{BatchOneHot, /* (OpSchema::CostInferenceFunctionType(CostInferenceForBatchOneHot)) */ }

tensor_inference_function!{BatchOneHot, /* (TensorInferenceForBatchOneHot) */}

value_key_length_input_fillers!{
    /*
    BatchOneHot, (
        BatchOneHotOp<CPUContext>::X,
        BatchOneHotOp<CPUContext>::VALS,
        BatchOneHotOp<CPUContext>::LENS
    )
    */
}

input_tags!{
    BatchOneHotOp {
        X,
        Lens,
        Vals
    }
}

output_tags!{
    BatchOneHotOp {
        OneHot
    }
}

impl<Context> BatchOneHotOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(X));
        */
    }
}
