crate::ix!();

///----------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct Col2ImOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    pad:         i32,
    kernel_h:    i32,
    kernel_w:    i32,
    dilation_h:  i32,
    dilation_w:  i32,
    stride_h:    i32,
    stride_w:    i32,
    order:       StorageOrder,

    phantom: PhantomData<T>,
}

num_inputs!{Col2Im, 2}

num_outputs!{Col2Im, 1}

impl<T, Context> Col2ImOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pad_(this->template GetSingleArgument<int>("pad", 0)),
            kernel_h_(this->template GetSingleArgument<int>(
                "kernel_h",
                this->template GetSingleArgument<int>("kernel", 0))),
            kernel_w_(this->template GetSingleArgument<int>(
                "kernel_w",
                this->template GetSingleArgument<int>("kernel", 0))),
            dilation_h_(this->template GetSingleArgument<int>(
                "dilation_h",
                this->template GetSingleArgument<int>("dilation", 1))),
            dilation_w_(this->template GetSingleArgument<int>(
                "dilation_w",
                this->template GetSingleArgument<int>("dilation", 1))),
            stride_h_(this->template GetSingleArgument<int>(
                "stride_h",
                this->template GetSingleArgument<int>("stride", 1))),
            stride_w_(this->template GetSingleArgument<int>(
                "stride_w",
                this->template GetSingleArgument<int>("stride", 1))),
            order_(StringToStorageOrder(
                this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE(kernel_h_ > 0);
        CAFFE_ENFORCE(kernel_w_ > 0);
        CAFFE_ENFORCE(dilation_h_ > 0);
        CAFFE_ENFORCE(dilation_w_ > 0);
        CAFFE_ENFORCE(stride_h_ > 0);
        CAFFE_ENFORCE(stride_w_ > 0);
        CAFFE_ENFORCE(pad_ >= 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
        auto& Z = Input(1);

        auto* Y = Output(0, Z.sizes(), at::dtype<T>());
        CAFFE_ENFORCE(4 == Y->dim());

        int N = 0, C = 0, H = 0, W = 0;
        switch (order_) {
          case StorageOrder::NCHW:
            N = Y->dim32(0);
            C = Y->dim32(1);
            H = Y->dim32(2);
            W = Y->dim32(3);
            break;
          case StorageOrder::NHWC:
            N = Y->dim32(0);
            H = Y->dim32(1);
            W = Y->dim32(2);
            C = Y->dim32(3);
            break;
          default:
            CAFFE_THROW("Unknown storage order: ", order_);
        }

        const int dkernel_h = dilation_h_ * (kernel_h_ - 1) + 1;
        const int dkernel_w = dilation_w_ * (kernel_w_ - 1) + 1;
        CAFFE_ENFORCE(H >= dkernel_h);
        CAFFE_ENFORCE(W >= dkernel_w);
        const int out_h = (H + 2 * pad_ - dkernel_h) / stride_h_ + 1;
        const int out_w = (W + 2 * pad_ - dkernel_w) / stride_w_ + 1;
        CAFFE_ENFORCE(X.numel() == N * kernel_h_ * kernel_w_ * C * out_h * out_w);

        const size_t dx = X.numel() / N;
        const size_t dy = Y->numel() / N;

        // could template-specialize this, but it's test code...
        switch (order_) {
          case StorageOrder::NCHW: {
            for (int n = 0; n < N; ++n) {
              const auto* xdata = X.template data<T>() + (n * dx);
              auto* ydata = Y->template mutable_data<T>() + (n * dy);
              math::Col2Im<T, Context, StorageOrder::NCHW>(
                  C,
                  H,
                  W,
                  kernel_h_,
                  kernel_w_,
                  dilation_h_,
                  dilation_w_,
                  pad_,
                  pad_,
                  pad_,
                  pad_,
                  stride_h_,
                  stride_w_,
                  xdata,
                  ydata,
                  &context_);
            }
          }; break;
          case StorageOrder::NHWC: {
            for (int n = 0; n < N; ++n) {
              const auto* xdata = X.template data<T>() + (n * dx);
              auto* ydata = Y->template mutable_data<T>() + (n * dy);
              math::Col2Im<T, Context, StorageOrder::NHWC>(
                  C,
                  H,
                  W,
                  kernel_h_,
                  kernel_w_,
                  dilation_h_,
                  dilation_w_,
                  pad_,
                  pad_,
                  pad_,
                  pad_,
                  stride_h_,
                  stride_w_,
                  xdata,
                  ydata,
                  &context_);
            }
          }; break;
          default:
            CAFFE_THROW("Unknown storage order: ", order_);
        }

        return true;
        */
    }
}
