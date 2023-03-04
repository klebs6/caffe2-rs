crate::ix!();

/**
  | Reshape the input tensor similar to
  | numpy.reshape.
  | 
  | It takes a tensor as input and an optional
  | tensor specifying the new shape.
  | 
  | When the second input is absent, an extra
  | argument `shape` must be specified.
  | 
  | It outputs the reshaped tensor as well
  | as the original shape.
  | 
  | At most one dimension of the new shape
  | can be -1.
  | 
  | In this case, the value is inferred from
  | the size of the tensor and the remaining
  | dimensions.
  | 
  | A dimension could also be 0, in which
  | case the actual dimension value is going
  | to be copied from the input tensor.
  |
  */
pub struct Int8ReshapeOp {
    base: ReshapeOp::<u8, CPUContext>,
} 

register_cpu_operator!{Int8Reshape, int8::Int8ReshapeOp}

num_inputs!{Int8Reshape, (1,2)}

num_outputs!{Int8Reshape, 2}

inputs!{Int8Reshape, 
    0 => ("data",      "An input tensor."),
    1 => ("new_shape", "New shape.")
}

outputs!{Int8Reshape, 
    0 => ("reshaped",  "Reshaped data."),
    1 => ("old_shape", "Original shape.")
}

args!{Int8Reshape, 
    0 => ("shape",        "New shape"),
    1 => ("Y_scale",      "Output tensor quantization scale"),
    2 => ("Y_zero_point", "Output tensor quantization offset")
}

allow_inplace!{Int8Reshape, vec![(0, 0)]}

impl Int8ReshapeOp {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ReshapeOp(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 2) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        }
        CAFFE_ENFORCE(
            OperatorStorage::HasArgument("shape"), "Argument `shape` is missing.");
        return this->template DoRunWithType<int64_t>();
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& X = Inputs()[0]->Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;
        DoRunWithTypeImpl<T>(X.t, &Y->t);
        return true;
        */
    }
}
