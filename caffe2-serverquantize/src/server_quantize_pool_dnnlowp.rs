crate::ix!();

use crate::{
    AveragePoolFp320p,
    PoolOp,
    MaxPoolFp320p,
    ConvPoolDNNLowPOpBase,
    CPUContext,
    OperatorDef,
    Workspace,
    AveragePoolFunctor,
    MaxPoolFunctor,
};

pub struct AveragePool<T> { 
    phantom: PhantomData<T>,
}

impl<T> AveragePool<T> {
    
    #[inline] pub fn initialize() -> f32 {
        
        todo!();
        /*
            return 0.0;
        */
    }
    
    #[inline] pub fn process_xdata_ydata(x_data: &T, y_data: &mut T)  {
        
        todo!();
        /*
            y_data += x_data;
        */
    }
    
    #[inline] pub fn finalize(size: i32, y_data: &mut T)  {
        
        todo!();
        /*
            y_data /= size;
        */
    }
    
    #[inline] pub fn process(
        x_col: i32,
        y_col: i32,
        x_mat: &mut ndarray::ArrayView2<f32>,
        y_mat: &mut ndarray::ArrayViewMut2<f32>)  
    {
        todo!();
        /*
            y_mat.col(y_col) += x_mat.col(x_col);
        */
    }
}

pub struct MaxPool<T> { 
    phantom: PhantomData<T>,
}

impl<T> MaxPool<T> {
    
    #[inline] pub fn initialize() -> T {
        
        todo!();
        /*
            return std::numeric_limits<T>::lowest();
        */
    }
    
    #[inline] pub fn process(
        x_col: i32,
        y_col: i32,
        x_mat: &mut ndarray::ArrayView2<f32>,
        y_mat: &mut ndarray::ArrayViewMut2<f32>)  
    {
        todo!();
        /*
            y_mat.col(y_col) = y_mat.col(y_col).cwiseMax(x_mat.col(x_col));
        */
    }
    
    #[inline] pub fn process_xdata_ydata(x_data: &T, y_data: &mut T)  {
        
        todo!();
        /*
            if (x_data > y_data) {
          y_data = x_data;
        }
        */
    }
    
    #[inline] pub fn finalize(size: i32, y_data: &mut T)  {
        
        todo!();
        /*
        
        */
    }
}

pub type AveragePoolFp32Op 
= PoolOp<f32, CPUContext, AveragePoolFunctor<CPUContext>>;

///------------------------------
pub struct AveragePoolDnnLowPOp<T: PrimInt> {
    //USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
    //USE_CONV_POOL_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, AveragePoolFp32Op);
    base: ConvPoolDNNLowPOpBase<T, AveragePoolFp320p>,
}

impl<T: PrimInt> AveragePoolDnnLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : BaseType(operator_def, ws) 

        for (int i = 0; i < this->kernel_.size(); ++i) {
          CAFFE_ENFORCE(
              dilation_[i] == 1, "Pooling op does not support dilation right now.");
        }
        if (!global_pooling_) {
          for (int i = 0; i < this->kernel_.size(); ++i) {
            CAFFE_ENFORCE(
                pads_[i] < kernel_[i] &&
                    pads_[i + this->kernel_.size()] < kernel_[i],
                "Pad should be smaller than kernel.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

        this->ParseDNNLowPOperatorArguments_();

        in_qparams_[0] =
            GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

        // Quantize input if needed
        vector<T> X_temp;
        const T* Xdata = QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);

        GetOutputQuantizationParams_();

        auto& X = InputTensorCPU_(0);
        auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, X.dim32(1));
        auto* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

        T* Ydata = GetQuantizedOutputData_();

        // The main loop
        int channels = X.dim32(1);
        int height = X.dim32(2);
        int width = this->kernel_.size() > 1 ? X.dim32(3) : 1;
        int depth = this->kernel_.size() > 2 ? X.dim32(4) : 1;
        int pooled_height = Y->dim32(2);
        int pooled_width = this->kernel_.size() > 1 ? Y->dim32(3) : 1;
        int pooled_depth = this->kernel_.size() > 2 ? Y->dim32(4) : 1;

        bool is_signed = std::is_signed<T>::value;
        int precision = out_qparams_.precision;
        int32_t minimum = is_signed ? -(1 << (precision - 1)) : 0;
        int32_t maximum =
            is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1;

        switch (this->kernel_.size()) {
          case 2:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                const T* Xdata_temp = Xdata + height * width * (c + channels * n);
                T* Ydata_temp =
                    Ydata + pooled_height * pooled_width * (c + channels * n);
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);

                    int size = (hend - hstart) * (wend - wstart);

                    const int pool_index = ph * pooled_width + pw;
                    int32_t Yh = -in_qparams_[0].zero_point * size;
                    for (int h = hstart; h < hend; ++h) {
                      for (int w = wstart; w < wend; ++w) {
                        const int input_index = h * width + w;
                        Yh += Xdata_temp[input_index];
                      }
                    }
                    float multiplier =
                        in_qparams_[0].scale / out_qparams_.scale / size;
                    Ydata_temp[pool_index] = std::min<int32_t>(
                        std::max<int32_t>(
                            nearbyint(Yh * multiplier + out_qparams_.zero_point),
                            minimum),
                        maximum);
                  } // width
                } // height
              } // channel
            } // for each image
            break;
          case 3:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                const T* Xdata_temp =
                    Xdata + height * width * depth * (c + channels * n);
                T* Ydata_temp = Ydata +
                    pooled_height * pooled_width * pooled_depth *
                        (c + channels * n);
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    for (int pd = 0; pd < pooled_depth; ++pd) {
                      int dstart = pd * stride_[2] - pads_[2];
                      int dend = min(dstart + kernel_[2], depth);
                      dstart = max(dstart, 0);

                      int size =
                          (hend - hstart) * (wend - wstart) * (dend - dstart);
                      const int pool_index =
                          ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                      int32_t Yh = -in_qparams_[0].zero_point * size;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          for (int d = dstart; d < dend; ++d) {
                            const int input_index =
                                h * width * depth + w * depth + d;
                            Yh += Xdata_temp[input_index];
                          }
                        }
                      }
                      float multiplier =
                          in_qparams_[0].scale / out_qparams_.scale / size;
                      Ydata_temp[pool_index] = std::min<int32_t>(
                          std::max<int32_t>(
                              nearbyint(Yh * multiplier + out_qparams_.zero_point),
                              minimum),
                          maximum);
                    } // depth
                  } // width
                } // height
                // Do offset.
              } // channel
            } // for each image
            break;
          default:
            CAFFE_THROW("Unsupported pooling size : ", this->kernel_.size());
            return false;
        }

        RunOnDeviceEpilogue_();
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            // average pooling
        using namespace dnnlowp;

        this->ParseDNNLowPOperatorArguments_();

        in_qparams_[0] =
            GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

        // Quantize input if needed
        vector<T> X_temp;
        const T* Xdata = QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);

        GetOutputQuantizationParams_();

        auto& X = InputTensorCPU_(0);
        int channels = X.dim32(X.ndim() - 1);
        auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, channels);
        auto* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

        T* Ydata = GetQuantizedOutputData_();

        int height = X.dim32(1);
        int width = this->kernel_.size() > 1 ? X.dim32(2) : 1;
        int depth = this->kernel_.size() > 2 ? X.dim32(3) : 1;
        int pooled_height = Y->dim32(1);
        int pooled_width = this->kernel_.size() > 1 ? Y->dim32(2) : 1;
        int pooled_depth = this->kernel_.size() > 2 ? Y->dim32(3) : 1;

        bool is_signed = std::is_signed<T>::value;
        int precision = out_qparams_.precision;
        int32_t minimum = is_signed ? -(1 << (precision - 1)) : 0;
        int32_t maximum =
            is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1;

        switch (this->kernel_.size()) {
          case 2:
            if (is_same<T, uint8_t>::value) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                average_pool_avx2(
                    reinterpret_cast<const uint8_t*>(Xdata),
                    n,
                    height,
                    width,
                    channels,
                    pooled_height,
                    pooled_width,
                    kernel_h(),
                    kernel_w(),
                    stride_h(),
                    stride_w(),
                    pad_t(),
                    pad_l(),
                    reinterpret_cast<uint8_t*>(Ydata),
                    in_qparams_[0].scale,
                    out_qparams_.scale,
                    in_qparams_[0].zero_point,
                    out_qparams_.zero_point,
                    minimum,
                    maximum);
              }
            } else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                const T* Xdata_temp = Xdata + n * height * width * channels;
                T* Ydata_temp = Ydata + n * pooled_height * pooled_width * channels;
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    int size = (hend - hstart) * (wend - wstart);
                    float multiplier =
                        in_qparams_[0].scale / out_qparams_.scale / size;

                    for (int c = 0; c < channels; ++c) {
                      const int pool_idx = (ph * pooled_width + pw) * channels + c;
                      int32_t Yh = -in_qparams_[0].zero_point * size;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int input_idx = (h * width + w) * channels + c;
                          Yh += Xdata_temp[input_idx];
                        }
                      }
                      Ydata_temp[pool_idx] = std::min<int32_t>(
                          std::max<int32_t>(
                              nearbyint(Yh * multiplier + out_qparams_.zero_point),
                              minimum),
                          maximum);
                    } // channel
                  } // width
                } // height
              } // for each image
            }
            break;
          case 3:
            if (is_same<T, uint8_t>::value) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                average_pool_3d_avx2(
                    reinterpret_cast<const uint8_t*>(Xdata),
                    n,
                    height,
                    width,
                    depth,
                    channels,
                    pooled_height,
                    pooled_width,
                    pooled_depth,
                    kernel_h(),
                    kernel_w(),
                    kernel_[2],
                    stride_h(),
                    stride_w(),
                    stride_[2],
                    pad_t(),
                    pad_l(),
                    pads_[2],
                    reinterpret_cast<uint8_t*>(Ydata),
                    in_qparams_[0].scale,
                    out_qparams_.scale,
                    in_qparams_[0].zero_point,
                    out_qparams_.zero_point,
                    minimum,
                    maximum);
              }
            } else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                const T* Xdata_temp = Xdata + n * height * width * depth * channels;
                T* Ydata_temp = Ydata +
                    n * pooled_height * pooled_width * pooled_depth * channels;
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    for (int pd = 0; pd < pooled_depth; ++pd) {
                      int dstart = pd * stride_[2] - pads_[2];
                      int dend = min(dstart + kernel_[2], depth);
                      dstart = max(dstart, 0);
                      int size =
                          (hend - hstart) * (wend - wstart) * (dend - dstart);
                      float multiplier =
                          in_qparams_[0].scale / out_qparams_.scale / size;

                      for (int c = 0; c < channels; ++c) {
                        const int pool_idx =
                            ((ph * pooled_width + pw) * pooled_depth + pd) *
                                channels +
                            c;
                        int32_t Yh = -in_qparams_[0].zero_point * size;
                        for (int h = hstart; h < hend; ++h) {
                          for (int w = wstart; w < wend; ++w) {
                            for (int d = dstart; d < dend; ++d) {
                              const int input_idx =
                                  ((h * width + w) * depth + d) * channels + c;
                              Yh += Xdata_temp[input_idx];
                            }
                          }
                        }
                        Ydata_temp[pool_idx] = std::min<int32_t>(
                            std::max<int32_t>(
                                nearbyint(
                                    Yh * multiplier + out_qparams_.zero_point),
                                minimum),
                            maximum);
                      } // channel
                    } // depth
                  } // width
                } // height
              } // for each image
            }
            break;
          default:
            CAFFE_THROW("Unsupported pooling size : ", this->kernel_.size());
            return false;
        }

        RunOnDeviceEpilogue_();
        return true;
        */
    }
}

pub type MaxPoolFp32Op = PoolOp<f32, CPUContext, MaxPoolFunctor<CPUContext>>;

///-----------------------------------------------
pub struct MaxPoolDnnLowPOp<T: PrimInt> {
    //USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
    //USE_CONV_POOL_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, MaxPoolFp32Op);
    base: ConvPoolDNNLowPOpBase<T, MaxPoolFp320p>,
}

impl<T: PrimInt> MaxPoolDnnLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : BaseType(operator_def, ws) 
        for (int i = 0; i < this->kernel_.size(); ++i) {
          CAFFE_ENFORCE(
              dilation_[i] == 1, "Pooling op does not support dilation right now.");
        }
        if (!global_pooling_) {
          for (int i = 0; i < this->kernel_.size(); ++i) {
            CAFFE_ENFORCE(
                pads_[i] < kernel_[i] &&
                    pads_[i + this->kernel_.size()] < kernel_[i],
                "Pad should be smaller than kernel.");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

        this->ParseDNNLowPOperatorArguments_();

        in_qparams_[0] =
            GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
        // Even if there is a pre-chosen quantization parameters for the output,
        // it is ignored because maxpool output quantization should be same as the
        // input.
        out_qparams_ = in_qparams_[0];

        // Quantize input if needed
        vector<T> X_temp;
        const T* Xdata = QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);

        auto& X = InputTensorCPU_(0);
        auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, X.dim32(1));
        auto* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

        T* Ydata = GetQuantizedOutputData_();

        // The main loop
        int channels = X.dim32(1);
        int height = X.dim32(2);
        int width = this->kernel_.size() > 1 ? X.dim32(3) : 1;
        int depth = this->kernel_.size() > 2 ? X.dim32(4) : 1;
        int pooled_height = Y->dim32(2);
        int pooled_width = this->kernel_.size() > 1 ? Y->dim32(3) : 1;
        int pooled_depth = this->kernel_.size() > 2 ? Y->dim32(4) : 1;

        switch (this->kernel_.size()) {
          case 1:
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  T Yh = MaxPool<T>::initialize();
                  for (int h = hstart; h < hend; ++h) {
                    MaxPool<T>::process(Xdata[h], Yh);
                  }
                  MaxPool<T>::finalize(hend - hstart, Yh);
                  Ydata[ph] = Yh;
                }
                // Do offset.
                Xdata += height;
                Ydata += pooled_height;
              }
            }
            break;
          case 2:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                // Do offset.
                const T* Xdata_temp = Xdata + height * width * (c + channels * n);
                T* Ydata_temp =
                    Ydata + pooled_height * pooled_width * (c + channels * n);
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    const int pool_index = ph * pooled_width + pw;
                    T Yh = MaxPool<T>::initialize();
                    for (int h = hstart; h < hend; ++h) {
                      for (int w = wstart; w < wend; ++w) {
                        const int input_index = h * width + w;
                        MaxPool<T>::process(Xdata_temp[input_index], Yh);
                      }
                    }
                    MaxPool<T>::finalize((hend - hstart) * (wend - wstart), Yh);
                    Ydata_temp[pool_index] = Yh;
                  }
                }
              }
            }
            break;
          case 3:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                // Do offset.
                const T* Xdata_temp =
                    Xdata + height * width * depth * (c + channels * n);
                T* Ydata_temp = Ydata +
                    pooled_height * pooled_width * pooled_depth *
                        (c + channels * n);
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    for (int pd = 0; pd < pooled_depth; ++pd) {
                      int dstart = pd * stride_[2] - pads_[2];
                      int dend = min(dstart + kernel_[2], depth);
                      dstart = max(dstart, 0);
                      const int pool_index =
                          ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                      T Yh = MaxPool<T>::initialize();
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          for (int d = dstart; d < dend; ++d) {
                            const int input_index =
                                h * width * depth + w * depth + d;
                            MaxPool<T>::process(Xdata_temp[input_index], Yh);
                          }
                        }
                      }
                      MaxPool<T>::finalize(
                          (hend - hstart) * (wend - wstart) * (dend - dstart), Yh);
                      Ydata_temp[pool_index] = Yh;
                    }
                  }
                }
              }
            }
            break;
          default:
            CAFFE_THROW("Unsupported pooling size : ", this->kernel_.size());
            return false;
        }

        if (measure_quantization_error_) {
          // to measure quantization error, run ref impl.
          Fp32Op_()->DequantizeInput();
          Fp32Op_()->Get()->RunOnDevice();
        }

        RunOnDeviceEpilogue_();
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            // max pooling
        using namespace dnnlowp;

        this->ParseDNNLowPOperatorArguments_();

        in_qparams_[0] =
            GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
        // Even if there is a pre-chosen quantization parameters for the output,
        // it is ignored because maxpool output quantization should be same as the
        // input.
        out_qparams_ = in_qparams_[0];

        // Quantize input if needed
        vector<T> X_temp;
        const T* Xdata = QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);

        auto& X = InputTensorCPU_(0);
        int channels = X.dim32(X.ndim() - 1);
        auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, channels);
        auto* Y = OutputTensorCPU_(0, sizes, at::dtype<T>());

        T* Ydata = GetQuantizedOutputData_();

        int height = X.dim32(1);
        int width = this->kernel_.size() > 1 ? X.dim32(2) : 1;
        int depth = this->kernel_.size() > 2 ? X.dim32(3) : 1;
        int pooled_height = Y->dim32(1);
        int pooled_width = this->kernel_.size() > 1 ? Y->dim32(2) : 1;
        int pooled_depth = this->kernel_.size() > 2 ? Y->dim32(3) : 1;

        switch (this->kernel_.size()) {
          case 1:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              const T* Xdata_temp = Xdata + n * height * channels;
              T* Ydata_temp = Ydata + n * pooled_height * channels;
              for (int ph = 0; ph < pooled_height; ++ph) {
                int hstart = ph * stride_h() - pad_t();
                int hend = min(hstart + kernel_h(), height);
                hstart = max(hstart, 0);
                for (int c = 0; c < channels; ++c) {
                  T Yh = MaxPool<T>::initialize();
                  const int pool_idx = ph * channels + c;
                  for (int h = hstart; h < hend; ++h) {
                    const int input_idx = h * channels + c;
                    MaxPool<T>::process(Xdata_temp[input_idx], Yh);
                  }
                  MaxPool<T>::finalize(hend - hstart, Yh);
                  Ydata_temp[pool_idx] = Yh;
                }
              }
            }
            break;
          case 2:
            if (is_same<T, uint8_t>::value) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                max_pool_avx2(
                    reinterpret_cast<const uint8_t*>(Xdata),
                    n,
                    height,
                    width,
                    channels,
                    pooled_height,
                    pooled_width,
                    kernel_h(),
                    kernel_w(),
                    stride_h(),
                    stride_w(),
                    pad_t(),
                    pad_l(),
                    reinterpret_cast<uint8_t*>(Ydata));
              }
            } else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
              for (int n = 0; n < X.dim32(0); ++n) {
                const T* Xdata_temp = Xdata + n * height * width * channels;
                T* Ydata_temp = Ydata + n * pooled_height * pooled_width * channels;
                for (int ph = 0; ph < pooled_height; ++ph) {
                  int hstart = ph * stride_h() - pad_t();
                  int hend = min(hstart + kernel_h(), height);
                  hstart = max(hstart, 0);
                  for (int pw = 0; pw < pooled_width; ++pw) {
                    int wstart = pw * stride_w() - pad_l();
                    int wend = min(wstart + kernel_w(), width);
                    wstart = max(wstart, 0);
                    int size = (hend - hstart) * (wend - wstart);
                    for (int c = 0; c < channels; ++c) {
                      T Yh = MaxPool<T>::initialize();
                      const int pool_idx = (ph * pooled_width + pw) * channels + c;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int input_idx = (h * width + w) * channels + c;
                          MaxPool<T>::process(Xdata_temp[input_idx], Yh);
                        }
                      }
                      MaxPool<T>::finalize(size, Yh);
                      Ydata_temp[pool_idx] = Yh;
                    }
                  }
                }
              }
            }
            break;
          case 3:
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
            for (int n = 0; n < X.dim32(0); ++n) {
              const T* Xdata_temp = Xdata + n * height * width * depth * channels;
              T* Ydata_temp = Ydata +
                  n * pooled_height * pooled_width * pooled_depth * channels;
              for (int ph = 0; ph < pooled_height; ++ph) {
                int hstart = ph * stride_h() - pad_t();
                int hend = min(hstart + kernel_h(), height);
                hstart = max(hstart, 0);
                for (int pw = 0; pw < pooled_width; ++pw) {
                  int wstart = pw * stride_w() - pad_l();
                  int wend = min(wstart + kernel_w(), width);
                  wstart = max(wstart, 0);
                  for (int pd = 0; pd < pooled_depth; ++pd) {
                    int dstart = pd * stride_[2] - pads_[2];
                    int dend = min(dstart + kernel_[2], depth);
                    dstart = max(dstart, 0);
                    int size = (hend - hstart) * (wend - wstart) * (dend - dstart);
                    for (int c = 0; c < channels; ++c) {
                      T Yh = MaxPool<T>::initialize();
                      const int pool_idx =
                          ((ph * pooled_width + pw) * pooled_depth + pd) *
                              channels +
                          c;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          for (int d = dstart; d < dend; ++d) {
                            const int input_idx =
                                ((h * width + w) * depth + d) * channels + c;
                            MaxPool<T>::process(Xdata_temp[input_idx], Yh);
                          }
                        }
                      }
                      MaxPool<T>::finalize(size, Yh);
                      Ydata_temp[pool_idx] = Yh;
                    }
                  }
                }
              }
            }
            break;
          default:
            CAFFE_THROW("Unsupported pooling size : ", this->kernel_.size());
            return false;
        }

        if (measure_quantization_error_) {
          // to measure quantization error, run ref impl.
          Fp32Op_()->DequantizeInput();
          Fp32Op_()->Get()->RunOnDevice();
        }

        RunOnDeviceEpilogue_();
        return true;
        */
    }
}

register_cpu_operator_with_engine!{
    AveragePool,
    DNNLOWP,
    AveragePoolDnnLowPOp<u8>
}

register_cpu_operator_with_engine!{
    MaxPool, 
    DNNLOWP, 
    MaxPoolDnnLowPOp<u8>
}

register_cpu_operator_with_engine!{
    AveragePool,
    DNNLOWP_16,
    AveragePoolDnnLowPOp<u16>
}

register_cpu_operator_with_engine!{
    MaxPool,
    DNNLOWP_16,
    MaxPoolDnnLowPOp<u16>
}

register_cpu_operator_with_engine!{
    Int8AveragePool,
    DNNLOWP,
    AveragePoolDnnLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8MaxPool,
    DNNLOWP,
    MaxPoolDnnLowPOp<u8>
}
