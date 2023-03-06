crate::ix!();

/**
  | The *InstanceNorm* op applies Instance
  | Normalization over a 4D input as described
  | in [Instance Normalization: The Missing
  | Ingredient for Fast Stylization] (https://arxiv.org/abs/1607.08022).
  | 
  | $$output = \frac{input-\mu_{input}}{\sqrt{\sigma_{input}^2}
  | + \epsilon}*scale + bias$$
  | 
  | Notice, two of the outputs are optional
  | so there are three output cases for this
  | op.
  | 
  | Case 1: output; Case 2: output, saved_mean;
  | Case 3: output, saved_mean, saved_inv_stdev.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc
  |
  */
pub struct InstanceNormOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    epsilon: f32,
    order:   StorageOrder,
    mean:    Tensor,
    rstd:    Tensor,
    scale:   Tensor,
    bias:    Tensor,

    phantom: PhantomData<T>,
}

register_cpu_operator!{
    InstanceNorm, 
    InstanceNormOp<f32, CPUContext>
}

allow_inplace!{InstanceNorm, vec![(0, 0)]}

num_inputs!{InstanceNorm, 3}

num_outputs!{InstanceNorm, (1,3)}

inputs!{InstanceNorm, 
    0 => ("input",            "The input 4-dimensional NCHW tensor to be operated on."),
    1 => ("scale",            "The input 1-dimensional scale tensor of size *C*."),
    2 => ("bias",             "The input 1-dimensional bias tensor of size *C*.")
}

outputs!{InstanceNorm, 
    0 => ("output",           "The output 4-dimensional tensor of the same shape as input."),
    1 => ("saved_mean",       "(Optional) Saved mean used during training to speed up gradient computation. Should not be used for testing."),
    2 => ("saved_inv_stdev",  "(Optional) Saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.")
}

args!{InstanceNorm, 
    0 => ("epsilon",          "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero."),
    1 => ("order",            "*(type: string; default: NCHW)* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is NHWC.")
}

input_tags!{
    InstanceNormOp {
        Input,
        Scale,
        Bias
    }
}

output_tags!{
    InstanceNormOp {
        Output,
        Mean,
        Rstd
    }
}

impl<T,Context> InstanceNormOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<string>("order", "NCHW"))) 

              CAFFE_ENFORCE_GE(epsilon_, 0, "Must pass a nonnegative epsilon.");
          CAFFE_ENFORCE_NE(
              order_,
              StorageOrder::UNKNOWN,
              "order should be either \"NCHW\" or \"NHWC\".");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
        const auto& gamma = Input(SCALE);
        const auto& beta = Input(BIAS);
        const int ndim = X.dim();
        const int64_t N = X.dim(0);
        const int64_t C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(ndim - 1);
        const int64_t HxW = X.numel() / (N * C);
        CAFFE_ENFORCE_EQ(gamma.numel(), C);
        CAFFE_ENFORCE_EQ(beta.numel(), C);
        auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
        const T* X_data = X.template data<T>();
        const T* gamma_data = gamma.template data<T>();
        const T* beta_data = beta.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        T* mean_data = nullptr;
        T* rstd_data = nullptr;
        if (OutputSize() >= 2) {
          auto* mean = Output(MEAN, {N, C}, at::dtype<T>());
          mean_data = mean->template mutable_data<T>();
        } else {
          ReinitializeTensor(
              &mean_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          mean_data = mean_.template mutable_data<T>();
        }
        if (OutputSize() >= 3) {
          auto* rstd = Output(RSTD, {N, C}, at::dtype<T>());
          rstd_data = rstd->template mutable_data<T>();
        } else {
          ReinitializeTensor(
              &rstd_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
          rstd_data = rstd_.template mutable_data<T>();
        }
        switch (order_) {
          case StorageOrder::NCHW: {
            return RunOnDeviceWithOrderNCHW(
                N,
                C,
                HxW,
                X_data,
                gamma_data,
                beta_data,
                Y_data,
                mean_data,
                rstd_data);
          }
          case StorageOrder::NHWC: {
            return RunOnDeviceWithOrderNHWC(
                N,
                C,
                HxW,
                X_data,
                gamma_data,
                beta_data,
                Y_data,
                mean_data,
                rstd_data);
          }
          default: {
            CAFFE_THROW("Unknown storage order: ", order_);
          }
        }
        */
    }
}
