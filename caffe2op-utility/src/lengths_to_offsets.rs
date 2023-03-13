crate::ix!();

/**
  | Given a vector of segment lengths, returns
  | a vector of offsets from these lengths,
  | which will have the same size as the input
  | vector.
  | 
  | Output is going to have the same type
  | as input.
  | 
  | For long tensors explicit casting from
  | int32 to int64 might be necessary prior
  | to this op.
  | 
  | For example, `[1, 3, 0, 2]` transforms
  | into `[0, 1, 4, 4]`.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LengthsToOffsetsOp<Context> {
    storage:             OperatorStorage,
    context:             Context,
    include_last_offset: bool,
}

num_inputs!{LengthsToOffsets, 1}

num_outputs!{LengthsToOffsets, 1}

inputs!{LengthsToOffsets, 
    0 => ("lengths", "1D tensor of int32 or int64 segment lengths.")
}

outputs!{LengthsToOffsets, 
    0 => ("offsets", "1D tensor of the same shape and type as `lengths`")
}

tensor_inference_function!{LengthsToOffsets, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          const ArgumentHelper args(def);
          bool include_last_offset =
              args.GetSingleArgument<bool>("include_last_offset", false);
          vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
          out_shape[0] += include_last_offset ? 1 : 0;
          return vector<TensorShape>{
              CreateTensorShape(out_shape, in[0].data_type())};
        */
    }
}

impl<Context> LengthsToOffsetsOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            include_last_offset_(this->template GetSingleArgument<bool>( "include_last_offset", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        auto* input_data = input.template data<int32_t>();

        CAFFE_ENFORCE(input.sizes().size() == 1, "Input must be a vector.");
        auto size = input.numel();

        output->Resize(size + (include_last_offset_ ? 1 : 0));
        auto* output_data = output->template mutable_data<int32_t>();

        int32_t offset = 0;
        for (int i = 0; i < size; ++i) {
          auto len = input_data[i];
          output_data[i] = offset;
          offset += len;
        }
        if (include_last_offset_) {
          output_data[size] = offset;
        }
        return true;
        */
    }
}
