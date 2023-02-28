crate::ix!();

/**
  | Given an input tensor in NCHW format,
  | the gradient for the output of SpatialBN
  | and the per-channel mean and inverse
  | std var vectors for the input, computes
  | the per-channel bias and scale gradient
  | to be used during the backward pass for
  | subsequent spatial batch normalization
  | gradient calculation.
  | 
  | Typically, the results of this op are
  | subsequently reduced over multiple
  | devices to obtain statistics over a
  | larger batch size in cases where the
  | batch size for a single model copy is
  | too low to yield the full benefit of batch
  | normalization. The resulting bias
  | and scale can then be plugged back into
  | SpatialBNGradient to get results over
  | the larger batch size
  |
  */
pub struct ChannelBackpropStatsOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    d_bias_scratch:  Tensor,
    d_scale_scratch: Tensor,
}

impl<Context> ChannelBackpropStatsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

input_tags!{
    ChannelBackpropStatsOp {
        Input,
        SavedMean,
        SavedInvStddev,
        OutputGrad
    }
}

output_tags!{
    ChannelBackpropStatsOp {
        ScaleGrad,
        BiasGrad
    }
}

impl ChannelBackpropStatsOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& dY = Input(OUTPUT_GRAD);
      CAFFE_ENFORCE(X.dim() >= 3 && X.dim() <= 5);
      const int N = X.dim32(0);
      const int C = X.dim32(1);
      const int H = X.dim32(2);
      const int W = X.dim() > 3 ? X.dim32(3) : 1;
      const int D = X.dim() > 4 ? X.dim32(4) : 1;

      const int sampleSize = H * W * D;

      Output(SCALE_GRAD)->Resize(C);
      Output(BIAS_GRAD)->Resize(C);
      auto* dScale = Output(SCALE_GRAD);
      auto* dBias = Output(BIAS_GRAD);

      ConstEigenArrayMap<float> X_arr(X.data<float>(), sampleSize, N * C);
      ConstEigenArrayMap<float> dY_arr(dY.data<float>(), sampleSize, N * C);
      ConstEigenVectorArrayMap<float> mean_arr(Input(SAVED_MEAN).data<float>(), C);
      ConstEigenVectorArrayMap<float> inv_stddev_arr(
          Input(SAVED_INV_STDDEV).data<float>(), C);
      EigenVectorArrayMap<float> dBias_arr(
          dBias->template mutable_data<float>(), C);
      EigenVectorArrayMap<float> dScale_arr(
          dScale->template mutable_data<float>(), C);

      dBias_arr.setZero();
      dScale_arr.setZero();

      for (int nc = 0; nc < N * C; ++nc) {
        int c = nc % C;
        dBias_arr(c) += dY_arr.col(nc).sum();
        dScale_arr(c) +=
            ((X_arr.col(nc) - mean_arr(c)) * inv_stddev_arr(c) * dY_arr.col(nc))
                .sum();
      }
      return true;
        */
    }
}

register_cpu_operator!{
    ChannelBackpropStats, 
    ChannelBackpropStatsOp<CPUContext>
}

num_inputs!{ChannelBackpropStats, 4}

num_outputs!{ChannelBackpropStats, 2}

inputs!{ChannelBackpropStats, 
    0 => ("X",            "The input 4-dimensional tensor of shape NCHW"),
    1 => ("mean",         "The mean saved from the forward pass as a 1-dimensional tensor of size C."),
    2 => ("inv_std",      "The saved inverse standard deviation as a 1-dimensional tensor of size C."),
    3 => ("output_grad",  "Gradient for the output layer of SpatialBN, here used as input because we are on the backward pass")
}

outputs!{ChannelBackpropStats, 
    0 => ("scale_grad",   "Gradient for the scale vector"),
    1 => ("bias_grad",    "Gradient for the bias vector")
}

should_not_do_gradient!{ChannelBackpropStats}
