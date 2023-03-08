crate::ix!();

/**
  | PadImage pads values around the boundary
  | of an image according to the pad values
  | and stride sizes defined by the ConvPoolOpBase
  | operator.
  |
  */
#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct PadImageOp<T,Context> {

    base: ConvPoolOpBase<Context>,

    mode:  PadMode,
    value: T,

    /*
      | Input: X
      | 
      | Output: Y
      |
      */
}

num_inputs!{PadImage, 1}

num_outputs!{PadImage, 1}

inputs!{PadImage, 
    0 => ("X", "Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. ")
}

outputs!{PadImage, 
    0 => ("Y", "Output data tensor from padding the H and W dimensions on the tensor. Dimensions will vary based on various pad and stride sizes.")
}

tensor_inference_function!{PadImage, /* (PadImageOp<float, CPUContext>::PadTensorInference) */}

impl<T,Context> PadImageOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
            mode_(StringToPadMode(
                this->template GetSingleArgument<string>("mode", "constant"))),
            value_(static_cast<T>(
                this->template GetSingleArgument<float>("value", 0.0))) 

        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::NOTSET,
            "Padding layer only supports explicit pad values.");
        CAFFE_ENFORCE(
            dilation_h() == 1 && dilation_w() == 1,
            "Pooling op does not support dilation right now.");
        CAFFE_ENFORCE(
            stride_h() == 1 && stride_w() == 1,
            "Pooling op does not support stride right now.");
        // Pad op does not use kernel sizes, so we set it to 1 for computing the
        // output size.
        kernel_.assign(pads_.size() / 2, 1);
        */
    }
}
