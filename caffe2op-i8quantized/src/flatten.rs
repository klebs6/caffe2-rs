crate::ix!();

/**
  | Flattens the input tensor into a 2D matrix.
  | If input tensor has shape
  | 
  | (d_0, d_1, ... d_n)
  | 
  | then the output will have shape
  | 
  | (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1)
  | ... X dn)
  |
  */
pub struct Int8FlattenOp {
    storage: OperatorStorage,
    context: CPUContext,
    axis:    i32,
}

register_cpu_operator!{Int8Flatten, int8::Int8FlattenOp}

num_inputs!{Int8Flatten, 1}

num_outputs!{Int8Flatten, 1}

inputs!{Int8Flatten, 
    0 => ("input", "A Int8 tensor of rank >= axis.")
}

outputs!{Int8Flatten, 
    0 => ("output", "A 2D Int8 tensor with the contents of the input tensor, with input dimensions up to axis flattened to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.")
}

args!{Int8Flatten, 
    0 => ("Y_scale", "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset"),
    2 => ("axis", "(Default to 1) Indicate up to which input dimensions (exclusive) should be flattened to the outer dimension of the output")
}

tensor_inference_function!{Int8Flatten, TensorInferenceForFlatten}

impl Int8FlattenOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Inputs()[0]->Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;
        CAFFE_ENFORCE_GE(
            X.t.sizes().size(), axis_, "The rank of the tensor must be >= axis.");
        Y->t.Resize(X.t.size_to_dim(axis_), X.t.size_from_dim(axis_));
        context_.CopyItemsToCPU(
            X.t.dtype(),
            X.t.numel(),
            X.t.raw_data(),
            Y->t.raw_mutable_data(X.t.dtype()));
        return true;
        */
    }
}
