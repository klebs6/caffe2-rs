crate::ix!();

use crate::{
    DNNLowPOp,
    ChannelShuffleOp,
    CPUContext,
    StorageOrder,
    OperatorDef,
    Workspace
};

pub type ChannelShuffleFp32Op<Context> = ChannelShuffleOp<f32, Context>;

pub struct ChannelShuffleDNNLowPOp<T> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ChannelShuffleFp32Op<CPUContext>);
    base: DNNLowPOp<T, ChannelShuffleFp32Op<CPUContext>>,

    order: StorageOrder,
    group: i32,
}

impl<T> ChannelShuffleDNNLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : BaseType(operator_def, ws),
          order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))),
          OP_SINGLE_ARG(int, "group", group_, 1) 

      CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                          : RunOnDeviceWithOrderNHWC();
        */
    }

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      this->ParseDNNLowPOperatorArguments_();

      // Choose quantization params
      TensorQuantizationParams in_qparams =
          GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      const auto& X = InputTensorCPU_(0);
      auto* Y = OutputTensorCPU_(0);
      Y->ResizeLike(X);
      const int N = X.dim32(0);
      const int C = X.dim32(1);
      const int G = group_;
      CAFFE_ENFORCE_EQ(C % G, 0);
      const int K = C / G;
      const int HxW = X.size_from_dim(2);
      const int stride = C * HxW;
      const T* X_data = X.template data<T>();
      T* Y_data = Y->template mutable_data<T>();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int i = 0; i < N; ++i) {
        ConstEigenMatrixMap<T> X_mat(X_data + i * stride, K * HxW, G);
        for (int j = 0; j < K; ++j) {
          EigenMatrixMap<T>(Y_data + i * stride + j * G * HxW, HxW, G) =
              X_mat.block(j * HxW, 0, HxW, G);
        }
      }

      // Even if there is a pre-chosen quantization parameters for the output,
      // it is ignored because channel shuffle output quantization should be same
      // as the input.
      PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      this->ParseDNNLowPOperatorArguments_();

      // Choose quantization params
      TensorQuantizationParams in_qparams =
          GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      const auto& X = InputTensorCPU_(0);
      auto* Y = OutputTensorCPU_(0);
      Y->ResizeLike(X);
      const auto C = X.dim32(X.ndim() - 1);
      const auto G = this->group_;
      CAFFE_ENFORCE(C % G == 0, "");
      const auto K = C / G;
      std::array<int, 2> dims = {G, K};
      std::array<int, 2> axes = {1, 0};
      const T* X_data = X.template data<T>();
      T* Y_data = Y->template mutable_data<T>();

      if (G == 4 && std::is_same<T, std::uint8_t>::value && GetCpuId().avx2()) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
        for (auto i = 0; i < X.numel(); i += C) {
          // Transpose each C = GxK matrix
          fbgemm::transpose_4rows(
              K,
              reinterpret_cast<const std::uint8_t*>(X_data + i),
              reinterpret_cast<std::uint8_t*>(Y_data + i));
        }
      } else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
        for (auto i = 0; i < X.numel(); i += C) {
          // Transpose each C = GxK matrix
          math::Transpose(
              2, dims.data(), axes.data(), X_data + i, Y_data + i, &context_);
        }
      }

      // Even if there is a pre-chosen quantization parameters for the output,
      // it is ignored because channel shuffle output quantization should be same
      // as the input.
      PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

      return true;
        */
    }
}

register_cpu_operator_with_engine!{
    ChannelShuffle,
    DNNLOWP,
    ChannelShuffleDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8ChannelShuffle,
    DNNLOWP,
    ChannelShuffleDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    ChannelShuffle,
    DNNLOWP_16,
    ChannelShuffleDNNLowPOp<u16>
}
