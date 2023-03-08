crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NCHW2NHWCOp<T,Context> {
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

register_cpu_operator!{
    NCHW2NHWC, 
    NCHW2NHWCOp<f32, CPUContext>
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
