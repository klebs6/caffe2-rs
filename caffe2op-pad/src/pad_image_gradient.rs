crate::ix!();

///-----------------------------------
#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct PadImageGradientOp<T,Context> {

    base: ConvPoolOpBase<Context>,

    mode: PadMode,

    /**
      | Input: dY
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{PadImageGradient, 1}

num_outputs!{PadImageGradient, 1}

impl<T,Context> PadImageGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
            mode_(StringToPadMode(this->template GetSingleArgument<string>("mode", "constant"))) 

        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::NOTSET,
            "Padding layer only supports explicit pad values.");
        CAFFE_ENFORCE(
            dilation_h() == 1 && dilation_w() == 1,
            "Pooling op does not support dilation right now.");
        // Pad op does not use kernel sizes, so we set it to 1 for computing the
        // output size.
        kernel_.assign(pads_.size() / 2, 1);
        */
    }
}
