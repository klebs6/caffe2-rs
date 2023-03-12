crate::ix!();

/**
  | The *LengthsRangeFill* op takes a single
  | input lengths* and outputs a single
  | tensor range_sequence*. For each element
  | of *lengths*, the op appends the range(0,lengths)
  | vector to the end of *range_sequence*.
  | For example, if input=[2,4,1], the
  | output would be [0,1,0,1,2,3,0].
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsRangeFillOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{LengthsRangeFill, 1}

num_outputs!{LengthsRangeFill, 1}

inputs!{LengthsRangeFill, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsRangeFill, 
    0 => ("range_sequence", "1D tensor whose size is the sum of *lengths*")
}

impl<Context> LengthsRangeFillOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);

        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE_EQ(input.dim(), 1, "Input must be a vector.");

        auto len_sum = std::accumulate(input_data, input_data + input.numel(), 0);

        auto* output = Output(0, {len_sum}, at::dtype<int32_t>());
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < input.numel(); ++i) {
          auto len = input_data[i];
          auto start = output_data + offset;
          std::iota(
              start,
              start + len,
              0); // make the third argument the arg of this operator
          offset += len;
        }
        return true;
        */
    }
}
