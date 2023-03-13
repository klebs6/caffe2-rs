crate::ix!();

/**
  | Given a vector of segment lengths, calculates
  | offsets of each segment and packs them
  | next to the lengths.
  | 
  | For the input vector of length N the output
  | is a Nx2 matrix with (offset, lengths)
  | packaged for each segment.
  | 
  | For example, `[1, 3, 0, 2]` transforms
  | into `[[0, 1], [1, 3], [4, 0], [4, 2]]`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToRangesOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{LengthsToRanges, 1}

num_outputs!{LengthsToRanges, 1}

inputs!{LengthsToRanges, 
    0 => ("lengths", "1D tensor of int32 segment lengths.")
}

outputs!{LengthsToRanges, 
    0 => ("ranges", "2D tensor of shape len(lengths) X 2 and the same type as `lengths`")
}

tensor_inference_function!{LengthsToRanges, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
          out_shape.push_back(2);
          return vector<TensorShape>{ 
              CreateTensorShape(out_shape, in[0].data_type())};
        */
    }
}

impl<Context> LengthsToRangesOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto size = input.numel();

        output->Resize(size, 2);
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < size; ++i) {
          auto len = input_data[i];
          output_data[i * 2] = offset;
          output_data[i * 2 + 1] = len;
          offset += len;
        }
        return true;
        */
    }
}
