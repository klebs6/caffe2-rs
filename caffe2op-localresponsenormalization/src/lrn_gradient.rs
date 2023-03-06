crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LRNGradientOp<T, Context> {

    base:                LRNOpBase<T, Context>,
    scale_:              *mut Tensor, // = nullptr;
    local_scale_tensor_: Tensor,      //{Context::GetDeviceType()};

    /**
      | Input: X, Y, scale, dY;
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{LRNGradient, 3}

num_outputs!{LRNGradient, 1}

input_tags!{
    LRNGradientOp {
        Input,
        Output,
        Scale,
        OutputGrad
    }
}

impl<T,Context> LRNGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : LRNOpBase<T, Context>(std::forward<Args>(args)...)
        */
    }
}

