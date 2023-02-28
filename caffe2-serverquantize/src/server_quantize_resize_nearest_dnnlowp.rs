crate::ix!();

use crate::{
    DNNLowPOp,
    ResizeNearestOp,
    Workspace,
    CPUContext,
    OperatorDef
};

pub type ResizeNearestFP32Op = ResizeNearestOp<f32,CPUContext>;

pub struct ResizeNearestDNNLowPOp<T> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ResizeNearestFP32Op);
    base: DNNLowPOp<T, ResizeNearestFP32Op>,

    width_scale:   f32,
    height_scale:  f32,
}

impl<T> ResizeNearestDNNLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : BaseType(operator_def, ws),
            width_scale_(this->template GetSingleArgument<float>("width_scale", 1)),
            height_scale_( this->template GetSingleArgument<float>("height_scale", 1)) 

        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);

        const auto& order = StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NHWC"));
        CAFFE_ENFORCE_EQ(order, StorageOrder::NHWC);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      this->ParseDNNLowPOperatorArguments_();

      // Choose quantization params
      in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      const auto& X = InputTensorCPU_(0);
      auto* Y = OutputTensorCPU_(0);

      CAFFE_ENFORCE_EQ(X.ndim(), 4);
      const int N = X.dim32(0);
      const int IH = X.dim32(1);
      const int IW = X.dim32(2);
      const int C = X.dim32(3);
      const int OW = IW * width_scale_;
      const int OH = IH * height_scale_;

      Y->Resize(N, OH, OW, C);
      const T* X_data = X.template data<T>();
      T* Y_data = Y->template mutable_data<T>();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int n = 0; n < N; ++n) {
        for (int y = 0; y < OH; ++y) {
          const int in_y = std::min((int)(y / height_scale_), (IH - 1));
          for (int x = 0; x < OW; ++x) {
            const int in_x = std::min((int)(x / width_scale_), (IW - 1));
            std::memcpy(
                &Y_data[((n * OH + y) * OW + x) * C],
                &X_data[((n * IH + in_y) * IW + in_x) * C],
                C * sizeof(T));
          }
        }
      }

      // Even if there is a pre-chosen quantization parameters for the output,
      // it is ignored because resize nearest output quantization should be same
      // as the input.
      PropagateOutputTensorQuantizationParams(this, 0, in_qparams_[0]);

      return true;
        */
    }
}

register_cpu_operator_with_engine!{
    Int8ResizeNearest,
    DNNLOWP,
    ResizeNearestDNNLowPOp<u8>
}
