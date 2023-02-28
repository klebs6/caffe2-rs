crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorStorage,
    OperatorDef,
};

/**
  | Note(Yangqing): I think it is possible to do
  | a more general swapaxes operator but I am
  | a little afraid of going down that general
  | path. Only implementing the two actually needed
  | ones here.
  */

/**
  | The operator switches the order of data
  | in a tensor from NHWC- sample index N,
  | height H, width H and channels C, to the
  | NCHW order (this is for 2D images).
  | 
  | In general, this operator switches
  | the order of data in a tensor from N H_1
  | ... H_k C to N C H_1 ... H_k for k-dimensional
  | features, and currently supports k=1,
  | 2, and 3.
  |
  */
pub struct NHWC2NCHWOp<T,Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{NHWC2NCHW, 1}

num_outputs!{NHWC2NCHW, 1}

inputs!{NHWC2NCHW, 
    0 => ("data", "The input data (Tensor) in the NHWC order.")
}

outputs!{NHWC2NCHW, 
    0 => ("output", "The output tensor (Tensor) in the NCHW order.")
}

tensor_inference_function!{NHWC2NCHW, /* ([](const OperatorDef& /*unused*/ /*def*/,
                                const std::vector<TensorShape>& in) {
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NHWC2NCHW must be >= 3 dimensional");
      std::vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims(0));
      out[0].add_dims(in[0].dims(in[0].dims_size() - 1));
      for (auto i = 1; i < in[0].dims_size() - 1; ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      return out;
    }) */
}

impl<T,Context> NHWC2NCHWOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        CAFFE_ENFORCE_GE(ndim, 3);
        const int N = X.dim32(0);
        const int C = X.dim32(ndim - 1);
        std::vector<int64_t> Y_dims(ndim);
        Y_dims[0] = N;
        Y_dims[1] = C;
        int HxW = 1;
        for (int i = 2; i < ndim; ++i) {
          Y_dims[i] = X.dim32(i - 1);
          HxW *= Y_dims[i];
        }
        auto* Y = Output(0, Y_dims, at::dtype<T>());
        if (X.numel() <= 0) {
          return true;
        }
        math::NHWC2NCHW<T, Context>(
            N,
            C,
            HxW,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

/**
  | The operator switches the order of data
  | in a tensor from NCHW- sample index N,
  | channels C, height H and width W, to the
  | NHWC order (this is for 2D images).
  | 
  | In general, this operator switches
  | the order of data in a tensor from N C H_1
  | ... H_k to N H_1 ... H_k C for k-dimensional
  | features, and currently supports k=1,
  | 2, and 3.
  |
  */
pub struct NCHW2NHWCOp<T,Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{NCHW2NHWC, 1}

num_outputs!{NCHW2NHWC, 1}

inputs!{NCHW2NHWC, 
    0 => ("data", "The input data (Tensor) in the NCHW order.")
}

outputs!{NCHW2NHWC, 
    0 => ("output", "The output tensor (Tensor) in the NHWC order.")
}

tensor_inference_function!{NCHW2NHWC, /* ([](const OperatorDef& /*unused*/ /*def*/,
                                const std::vector<TensorShape>& in) {
      CAFFE_ENFORCE_GE(
          in[0].dims_size(), 3, "Input for NCHW2NHWC must be >= 3 dimensional");
      std::vector<TensorShape> out(1);
      out[0].add_dims(in[0].dims(0));
      for (auto i = 2; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].add_dims(in[0].dims(1));
      return out;
    }) */
}

impl<T,Context> NCHW2NHWCOp<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        CAFFE_ENFORCE_GE(ndim, 3);
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        std::vector<int64_t> Y_dims(ndim);
        Y_dims[0] = N;
        Y_dims[ndim - 1] = C;
        int HxW = 1;
        for (int i = 1; i < ndim - 1; ++i) {
          Y_dims[i] = X.dim32(i + 1);
          HxW *= Y_dims[i];
        }
        auto* Y = Output(0, Y_dims, at::dtype<T>());
        if (X.numel() <= 0) {
          return true;
        }
        math::NCHW2NHWC<T, Context>(
            N,
            C,
            HxW,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

register_cpu_operator!{NHWC2NCHW, NHWC2NCHWOp<f32, CPUContext>}

register_cpu_operator!{NCHW2NHWC, NCHW2NHWCOp<f32, CPUContext>}

pub struct GetNHWC2NCHWGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNHWC2NCHWGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "NCHW2NHWC",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

pub struct GetNCHW2NHWCGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetNCHW2NHWCGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "NHWC2NCHW",
            "",
            std::vector<std::string>{GO(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{NHWC2NCHW, GetNHWC2NCHWGradient}

register_gradient!{NCHW2NHWC, GetNCHW2NHWCGradient}
