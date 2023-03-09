crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SoftmaxOp<T,Context> {
    storage:         OperatorStorage,
    context:         Context,
    axis:            i32,
    scale:           Tensor,
    rowmax:          Tensor,
    sum_multiplier:  Tensor,
    phantom:         PhantomData<T>,
}

num_inputs!{Softmax, 1}

num_outputs!{Softmax, 1}

inputs!{Softmax, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor that's coerced into a 2D matrix of size (NxD) as described above.")
}

outputs!{Softmax, 
    0 => ("Y", "*(type: Tensor`<float>`)* The softmax normalized output tensor with the same shape as input tensor.")
}

args!{Softmax, 
    0 => ("axis", "*(type: int; default: 1)* Axis of the inputs when coerced to 2D matrix.")
}

identical_type_and_shape!{Softmax}

inherit_onnx_schema!{Softmax}

register_cpu_operator!{
    Softmax, 
    SoftmaxOp<f32, CPUContext>
}

register_cpu_gradient_operator!{
    SoftmaxGradient, 
    SoftmaxGradientOp<f32, CPUContext>
}

impl<T,Context> SoftmaxOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
}
