crate::ix!();

/**
  | Given a vector of segment lengths (*lengths*)
  | the *LengthsToSegmentIds* op returns
  | a zero-based, consecutive vector of
  | segment ids (*segment_ids*).
  | 
  | For example, *lengths=[1, 3, 0, 2]*
  | will produce segment_ids=[0, 1, 1,
  | 1, 3, 3]*.
  | 
  | In general, the inverse operation is
  | *SegmentIdsToLengths*.
  | 
  | Notice though that trailing empty sequence
  | lengths can't be properly recovered
  | from segment ids.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToSegmentIdsOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

disallow_input_fillers!{LengthsToSegmentIdsOp}

num_inputs!{LengthsToSegmentIds, 1}

num_outputs!{LengthsToSegmentIds, 1}

inputs!{LengthsToSegmentIds, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsToSegmentIds, 
    0 => ("segment_ids", "1D tensor of length *sum(lengths)*")
}

impl<Context> LengthsToSegmentIdsOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto total_length =
            std::accumulate(input_data, input_data + input.numel(), 0);

        output->Resize(total_length);
        auto* output_data = output->template mutable_data<int32_t>();

        for (int i = 0; i < input.numel(); ++i) {
          auto len = input_data[i];
          std::fill(output_data, output_data + len, i);
          output_data += len;
        }
        return true;
        */
    }
}
