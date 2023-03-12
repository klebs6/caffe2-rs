crate::ix!();

#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct ConvGradientOp<T, Context> {
    base:                    ConvPoolOpBase<Context>,
    col_buffer:              Tensor,
    bias_multiplier:         Tensor,
    img_shape_device:        Tensor, //{Context::GetDeviceType()};
    col_buffer_shape_device: Tensor, //{Context::GetDeviceType()};
    no_bias:                 bool,
    phantom:                 PhantomData<T>,

    // input: X, W, dY
    // output: dW, db, and optionally dX
}

input_tags!{
    ConvGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    ConvGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T,Context> ConvGradientOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws),
            no_bias_(this->template GetSingleArgument<int>("no_bias", 0)) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        CAFFE_ENFORCE(
            (group_ == 1 || order_ == StorageOrder::NCHW ||
             std::is_same<Context, CPUContext>::value),
            "Group convolution only supports NCHW order or CPUContext right now.");
        */
    }
}

type doc_fn = fn(s: &OpSchema) -> ();
