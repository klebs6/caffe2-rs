crate::ix!();

/**
  | Produces tensor containing data of
  | first input and shape of second input.
  | 
  | Output gets the data of input(0), but
  | reshapes it like input(1).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ResizeLikeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

should_not_do_gradient!{ResizeLike}

num_inputs!{ResizeLike, 2}

num_outputs!{ResizeLike, 1}

inputs!{ResizeLike, 
    0 => ("data",         "Tensor whose data will be copied into the output."),
    1 => ("shape_tensor", "Tensor whose shape will be applied to output.")
}

outputs!{ResizeLike, 
    0 => ("output",       "Tensor with data of input 0 and shape of input 1.")
}

tensor_inference_function!{
    ResizeLike, 

    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          vector<TensorShape> out(1);
          out.at(0) = in.at(1);
          out.at(0).set_data_type(in.at(0).data_type());
          return out;
        */
    }
}

impl<Context> ResizeLikeOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input0 = Input(0);
        auto& input1 = Input(1);
        auto* output = Output(0);
        CAFFE_ENFORCE_EQ(input0.numel(), input1.numel());
        output->ResizeLike(Input(1));
        context_.CopyItemsSameDevice(
            input0.dtype(),
            input0.numel(),
            input0.raw_data(),
            output->raw_mutable_data(input0.dtype()));
        return true;
        */
    }
}
