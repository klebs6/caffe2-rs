crate::ix!();

#[USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS("Context")]
pub struct ConvTransposeGradientOp<T, Context> {
    base:            ConvTransposeUnpoolBase<Context>,
    col_buffer:      Tensor,
    bias_multiplier: Tensor,
    no_bias:         bool,

    // input: X, W, dY
    // output: dW, optionally db and dX
    phantom: PhantomData<T>,
}

input_tags!{
    ConvTransposeGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    ConvTransposeGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T,Context> ConvTransposeGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvTransposeUnpoolBase<Context>(std::forward<Args>(args)...),
            no_bias_(this->template GetSingleArgument<bool>("no_bias", false)) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        */
    }
}
