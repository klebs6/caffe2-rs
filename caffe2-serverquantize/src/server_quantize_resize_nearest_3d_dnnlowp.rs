crate::ix!();

use crate::{
    ResizeNearest3DOp,
    DNNLowPOp,
    Workspace,
    OperatorDef,
    CPUContext
};

pub type ResizeNearest3DFP32Op = ResizeNearest3DOp<f32,CPUContext>;

pub struct ResizeNearest3DDNNLowPOp<T> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ResizeNearest3DFP32Op);
    base: DNNLowPOp<T, ResizeNearest3DFP32Op>,

    temporal_scale:  f32,
    width_scale:     f32,
    height_scale:    f32,
}

impl<T> ResizeNearest3DDNNLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : BaseType(operator_def, ws),
            temporal_scale_( this->template GetSingleArgument<float>("temporal_scale", 1)),
            width_scale_(this->template GetSingleArgument<float>("width_scale", 1)),
            height_scale_( this->template GetSingleArgument<float>("height_scale", 1)) 

        CAFFE_ENFORCE_GT(temporal_scale_, 0);
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

      CAFFE_ENFORCE_EQ(X.ndim(), 5);
      const int N = X.dim32(0);
      // input frames
      const int IF = X.dim32(1);
      const int IH = X.dim32(2);
      const int IW = X.dim32(3);
      const int C = X.dim32(4);
      const int OF = IF * temporal_scale_;
      const int OH = IH * height_scale_;
      const int OW = IW * width_scale_;

      vector<int> buffer_shape{N, OF, OH, OW, C};
      Y->Resize(buffer_shape);
      const T* X_data = X.template data<T>();
      T* Y_data = Y->template mutable_data<T>();

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int n = 0; n < N; ++n) {
        for (int t = 0; t < OF; ++t) {
          const int in_f = std::min((int)(t / temporal_scale_), (IF - 1));
          for (int y = 0; y < OH; ++y) {
            const int in_y = std::min((int)(y / height_scale_), (IH - 1));
            for (int x = 0; x < OW; ++x) {
              const int in_x = std::min((int)(x / width_scale_), (IW - 1));
              std::memcpy(
                  &Y_data[((((n * OF) + t) * OH + y) * OW + x) * C],
                  &X_data[((((n * IF) + in_f) * IH + in_y) * IW + in_x) * C],
                  C * sizeof(T));
            }
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
    Int8ResizeNearest3D,
    DNNLOWP,
    ResizeNearest3DDNNLowPOp<u8>
}
