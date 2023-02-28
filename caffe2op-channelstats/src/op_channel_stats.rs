crate::ix!();

/**
  | Given an input tensor in NCHW format,
  | computes the sum of all elements per
  | channel and the sum of all elements squared
  | per channel. These values can be reduced
  | across multiple batches and used to
  | obtain the mean and variance across
  | the full set of batches. Using the new
  | mean and variance as input to SpatialBN
  | has the effect of changing the batch
  | size over which SpatialBN is applied.
  |
  */
pub struct ChannelStatsOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    order:   StorageOrder,
}

register_cpu_operator!{
    ChannelStats, 
    ChannelStatsOp<CPUContext>
}

num_inputs!{ChannelStats, 1}

num_outputs!{ChannelStats, 2}

inputs!{ChannelStats, 
    0 => ("X", "The input 4-dimensional tensor of shape NCHW")
}

outputs!{ChannelStats, 
    0 => ("sum",   "The output 1-dimensional tensor of size C containing the sum of elements of X per channel."),
    1 => ("sumsq", "The output 1-dimensional tensor of size C containing the sum of elements squared per channel.")
}

should_not_do_gradient!{ChannelStats}

impl<Context> ChannelStatsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<std::string>("order", "NCHW"))) 
        CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const int ndim = X.dim();
            const int N = X.dim32(0);
            const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
            const int HxW = X.numel() / (N * C);
            auto* sum = Output(0, {C}, at::dtype<T>());
            auto* sumsq = Output(1, {C}, at::dtype<T>());
            const T* X_data = X.template data<T>();
            T* sum_data = sum->template mutable_data<T>();
            T* sumsq_data = sumsq->template mutable_data<T>();
            return order_ == StorageOrder::NCHW
                ? ComputeChannelStatsNCHW<T>(N, C, HxW, X_data, sum_data, sumsq_data)
                : ComputeChannelStatsNHWC<T>(N, C, HxW, X_data, sum_data, sumsq_data);
        */
    }
}

impl ChannelStatsOp<CPUContext> {
    
    #[inline] pub fn compute_channel_statsNCHW<T: Float>(
        &mut self, 
        n:         i32,
        c:         i32,
        hxW:       i32,
        x:         *const f32,
        sum:       *mut f32,
        sumsq:     *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
      for (int i = 0; i < C; ++i) {
        sum[i] = X_arr.col(i).sum();
        sumsq[i] = X_arr.col(i).square().sum();
      }
      for (int i = 1; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
          const int c = i * C + j;
          sum[j] += X_arr.col(c).sum();
          sumsq[j] += X_arr.col(c).square().sum();
        }
      }
      return true;
        */
    }
    
    #[inline] pub fn compute_channel_statsNHWC(
        &mut self, 
        n:      i32,
        c:      i32,
        hxW:    i32,
        x:      *const f32,
        sum:    *mut f32,
        sumsq:  *mut f32) -> bool 
    {
        todo!();
        /*
            ConstEigenArrayMap<float> X_arr(X, C, N * HxW);
      EigenVectorArrayMap<float> sum_arr(sum, C);
      EigenVectorArrayMap<float> sumsq_arr(sumsq, C);
      sum_arr = X_arr.col(0);
      sumsq_arr = X_arr.col(0).square();
      for (int i = 1; i < N * HxW; ++i) {
        sum_arr += X_arr.col(i);
        sumsq_arr += X_arr.col(i).square();
      }
      return true;
        */
    }
}
