crate::ix!();

/**
  | The *FlattenToVec* op flattens the
  | input tensor into a 1-D vector.
  | 
  | The op accepts a single input tensor
  | and returns a single output tensor.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FlattenToVecOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{FlattenToVec, 1}

num_outputs!{FlattenToVec, 1}

inputs!{FlattenToVec, 
    0 => ("input", "A tensor of rank >= 1.")
}

outputs!{FlattenToVec, 
    0 => ("output", "A tensor of rank 1 (vector) with the contents of the input tensor.")
}

tensor_inference_function!{
    FlattenToVec, 
    /* [](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      int total = 1;
      for (auto d : in[0].dims()) {
        total *= d;
      }
      out[0].set_data_type(in[0].data_type());
      out[0].add_dims(total);
      return out;
    } */
}

impl<Context> FlattenToVecOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);
        CAFFE_ENFORCE_GE(input.dim(), 1, "The rank of the tensor must be >= 1.");
        output->Resize(input.numel());

        context_.CopyItemsSameDevice(
            input.dtype(),
            input.numel(),
            input.raw_data(),
            output->raw_mutable_data(input.dtype()));
        return true;
        */
    }
}
