crate::ix!();

/**
  | The convolution fusion operator consumes
  | an input vector, a {dim}filter blob,
  | a bias blob and another input vector
  | and computes the output.
  | 
  | This operator gives the chance to fuse
  | the ReLU or element-wise Sum with a convolution
  | operator.
  | 
  | -----------
  | @note
  | 
  | other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | ConvPoolOpBase operator.
  | 
  | Various dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator. As is expected,
  | the filter is convolved with a subset
  | of the image and the bias is added;
  | 
  | this is done throughout the image data
  | and the output is computed.
  | 
  | As a side note on the implementation
  | layout: conv_op_impl.h is the templated
  | implementation of the conv_op.h file,
  | which is why they are separate files.
  |
  */
#[USE_IDEEP_DEF_ALIASES]
#[USE_IDEEP_CONV_POOL_BASE_FUNCTIONS]
pub struct IDEEPConvFusionOp {
    base: IDEEPConvOp,
}

num_inputs!{ConvFusion, (2,4)}

num_outputs!{ConvFusion, 1}

inputs!{ConvFusion, 
    0 => ("X", "Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints. "),
    1 => ("filter", "The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel."),
    2 => ("bias", "The 1D bias blob that is added through the convolution; has size (M)."),
    3 => ("S", "Input data blob for element-wise Sum fusion from previous layer; has the same size of convolution output. Its input index should be 2 if no bias for this convolution, and it MUST be inplace with output Y.")
}

outputs!{ConvFusion, 
    0 => ("Y", "Output data blob that contains the result of the convolution fusion. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

args!{ConvFusion, 
    0 => ("fusion_type", "Which fusion type is used")
}

allow_inplace!{ConvFusion, vec![(2, 0), (3, 0)]}

tensor_inference_function!{ConvFusion, /* ConvPoolOpBase<CPUContext>::TensorInferenceForConv */}

cost_inference_function!{ConvFusion, /* OpSchema::CostInferenceFunctionType( ConvPoolOpBase<CPUContext>::CostInferenceForConv) */ }

impl IDEEPConvFusionOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : IDEEPConvOp(operator_def, ws) 

        CAFFE_ENFORCE(OperatorStorage::HasArgument("fusion_type"),
              "You should specify the fusion type");
        fusion_type_ = static_cast<FusionType>(
            OperatorStorage::GetSingleArgument<int>("fusion_type", FUSION_UNKNOWN));
        OPERATOR_NEEDS_FEATURE(
            fusion_type_ > FUSION_UNKNOWN && fusion_type_ < FUSION_MAX,
            "Undefined Conv fusion type.",
            fusion_type_);

        switch (fusion_type_) {
          case FUSION_CONV_RELU:
            attr_ = iattr::fuse_relu();
            last_input_ = BIAS_OR_INPUT_S;
            break;
          case FUSION_CONV_SUM:
            attr_ = iattr::fuse_sum();
            last_input_ = INPUT_S;
            break;
          case FUSION_CONV_SUM_RELU:
            attr_ = iattr::residual();
            last_input_ = INPUT_S;
            break;
          default:
            CAFFE_THROW("Unsupported conv fusion type!");
        }
        */
    }
}
