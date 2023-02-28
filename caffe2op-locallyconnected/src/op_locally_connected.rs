crate::ix!();

use crate::{
    OperatorDef,
    GradientMakerBase,
    Tensor,
    ConvPoolOpBase,
};

/**
  | The locally connected operator consumes
  | an input vector, a N-D filter blob and
  | a bias blob and computes the output.
  | 
  | -----------
  | @note
  | 
  | other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | ConvPoolOpBase operator. Various
  | dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator. As is expected,
  | the filter is locally connected with
  | a subset of the image and the bias is added;
  | this is done throughout the image data
  | and the output is computed. As a side
  | note on the implementation layout:
  | locally_connected_op_impl.h is the
  | templated implementation of the locally_connected_op.h
  | file, which is why they are separate
  | files.
  |
  */
pub struct LocallyConnectedOp<T, Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    bias_multiplier:          Tensor, //{Context::GetDeviceType()};

    // Buffer.
    column_buffer:            Tensor, //{Context::GetDeviceType()};
    column_transposed_buffer: Tensor, //{Context::GetDeviceType()};
    y_transposed_buffer:      Tensor, //{Context::GetDeviceType()};

    phantom:                  PhantomData<T>,
}

inputs!{LocallyConnected,
    1 => ("filter", "The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel." ),
    2 => ("bias",   "The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).")
}

outputs!{LocallyConnected,
    0 => ("Y", "Output data blob that contains the result of the locally connected op. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

impl<T,Context> LocallyConnectedOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...) 

        // Since this is the default locally connected implementation, we will
        // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
        CAFFE_ENFORCE(
            group_ == 1 || order_ == StorageOrder::NCHW,
            "Group locally connected only supports NCHW order right now.");
        */
    }
}

/**
  | Input: X, W, b
  | 
  | Output: Y
  |
  */
input_tags!{
    LocallyConnectedOp {
        Input,
        Filter,
        Bias
    }
}

///------------------------------------------------
pub struct LocallyConnectedGradientOp<T, Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base:    ConvPoolOpBase<Context>,

    no_bias: bool,

    // Buffer.
    column_buffer:            Tensor, //{Context::GetDeviceType()};
    column_transposed_buffer: Tensor, //{Context::GetDeviceType()};
    dY_transposed_buffer:     Tensor, //{Context::GetDeviceType()};
    bias_multiplier:          Tensor, //{Context::GetDeviceType()};

    phantom: PhantomData<T>,
}

/**
  | input: X, W, dY
  | 
  | output: dW, db, and optionally dX
  |
  */
input_tags!{
    LocallyConnectedGradientOp {
        Input,
        Filter,
        OutputGrad
    }
}

output_tags!{
    LocallyConnectedGradientOp {
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T, Context> LocallyConnectedGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "no_bias", no_bias_, false) 

        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 3),
            "If bias is not present, you should not have 3 grad output.");
        CAFFE_ENFORCE(
            group_ == 1 || order_ == StorageOrder::NCHW,
            "Group locally connected only supports NCHW order right now.");
        */
    }
}

///----------------------------
register_cpu_operator!{LC, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC, (2,3)}

num_outputs!{LC, 1}

tensor_inference_function!{LC, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC1D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC1D, (2,3)}

num_outputs!{LC1D, 1}

tensor_inference_function!{LC1D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC2D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC2D, (2,3)}

num_outputs!{LC2D, 1}

tensor_inference_function!{LC2D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LC3D, LocallyConnectedOp<f32, CPUContext>}

num_inputs!{LC3D, (2,3)}

num_outputs!{LC3D, 1}

tensor_inference_function!{LC3D, /* ConvPoolOpBase<CPUContext>::TensorInferenceForLC */}

///----------------------------
register_cpu_operator!{LCGradient, LocallyConnectedGradientOp<f32, CPUContext>}

num_inputs!{LCGradient, (2,3)}

num_outputs!{LCGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC1DGradient, LocallyConnectedGradientOp<f32, CPUContext>}

num_inputs!{LC1DGradient, (2,3)}

num_outputs!{LC1DGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC2DGradient, LocallyConnectedGradientOp<f32, CPUContext> }

num_inputs!{LC2DGradient, (2,3)}

num_outputs!{LC2DGradient, (1,3)}

///----------------------------
register_cpu_operator!{LC3DGradient, LocallyConnectedGradientOp<f32, CPUContext> }

num_inputs!{LC3DGradient, (2,3)}

num_outputs!{LC3DGradient, (1,3)}

///-----------------------------
pub struct GetLocallyConnectedGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetLocallyConnectedGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 2);
        ArgumentHelper argsHelper(def_);
        const bool compute_dX =
            !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

        if (def_.input_size() == 3) {
          if (compute_dX) {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(2), GI(0)});
          } else {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(2)});
          }
        } else {
          if (compute_dX) {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1), GI(0)},
                std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
          } else {
            return SingleGradientDef(
                def_.type() + "Gradient",
                "",
                std::vector<string>{I(0), I(1), GO(0)},
                std::vector<string>{GI(1)},
                std::vector<Argument>{MakeArgument<int>("no_bias", 1)});
          }
        }
        */
    }
}

register_gradient!{LC,   GetLocallyConnectedGradient}
register_gradient!{LC1D, GetLocallyConnectedGradient}
register_gradient!{LC2D, GetLocallyConnectedGradient}
register_gradient!{LC3D, GetLocallyConnectedGradient}

register_cuda_operator!{LC,             LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LCGradient,     LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC1D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC1DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC2D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC2DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}

register_cuda_operator!{LC3D,           LocallyConnectedOp<f32, CUDAContext>}
register_cuda_operator!{LC3DGradient,   LocallyConnectedGradientOp<f32, CUDAContext>}
