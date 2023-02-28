crate::ix!();

use crate::{
    CPUContext,
    ConvPoolOpBase,
    GradientMakerBase,
    OperatorDef,
    TensorShape,
};

/**
  | Padding mode similar to numpy.
  |
  */
pub enum PadMode {

    /**
      | pad constant values, with string "constant"
      |
      */
    CONSTANT = 0, 

    /**
      | pads with reflect values, with string
      | "reflect"
      |
      */
    REFLECT = 1,

    /**
      | pads with the edge values, with string
      | "edge"
      |
      */
    EDGE = 2,
}

/**
  | PadImage pads values around the boundary
  | of an image according to the pad values
  | and stride sizes defined by the ConvPoolOpBase
  | operator.
  |
  */
pub struct PadImageOp<T,Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    mode:  PadMode,
    value: T,

    /*
      | Input: X
      | 
      | Output: Y
      |
      */
}

num_inputs!{PadImage, 1}

num_outputs!{PadImage, 1}

inputs!{PadImage, 
    0 => ("X", "Input data tensor from the previous operator; dimensions depend on whether the NCHW or NHWC operators are being used. For example, in the former, the input has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. The corresponding permutation of dimensions is used in the latter case. ")
}

outputs!{PadImage, 
    0 => ("Y", "Output data tensor from padding the H and W dimensions on the tensor. Dimensions will vary based on various pad and stride sizes.")
}

tensor_inference_function!{PadImage, /* (PadImageOp<float, CPUContext>::PadTensorInference) */}

impl<T,Context> PadImageOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
            mode_(StringToPadMode(
                this->template GetSingleArgument<string>("mode", "constant"))),
            value_(static_cast<T>(
                this->template GetSingleArgument<float>("value", 0.0))) 

        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::NOTSET,
            "Padding layer only supports explicit pad values.");
        CAFFE_ENFORCE(
            dilation_h() == 1 && dilation_w() == 1,
            "Pooling op does not support dilation right now.");
        CAFFE_ENFORCE(
            stride_h() == 1 && stride_w() == 1,
            "Pooling op does not support stride right now.");
        // Pad op does not use kernel sizes, so we set it to 1 for computing the
        // output size.
        kernel_.assign(pads_.size() / 2, 1);
        */
    }
}

impl PadImageOp<f32, CPUContext> {

    #[inline] pub fn pad_tensor_inference(
        &mut self, 
        def:   &OperatorDef, 
        input: &Vec<TensorShape>) -> Vec<TensorShape> {
        
        todo!();
        /*
            return ConvPoolOpBase::TensorInferenceForPool(def, in);
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      int channels = X.dim32(1);
      int height = X.dim32(2);
      int width = X.dim32(3);
      ConvPoolOpBase::SetOutputSize(X, Y, channels);

      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      // The main loop
      int padded_height = Y->dim32(2);
      int padded_width = Y->dim32(3);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  Ydata[ph * padded_width + pw] =
                      (h < 0 || w < 0 || h >= height || w >= width)
                      ? value_
                      : Xdata[h * width + w];
                }
              }
              // Do offset.
              Xdata += height * width;
              Ydata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::REFLECT:
          if (pad_r() >= 0 && pad_t() >= 0 && pad_l() >= 0 && pad_b() >= 0) {
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                // Handle the valid region:
                // i.e. Y[n][c][pad_t:pad_t+h][pad_l:pad_l+w]
                auto* Ystart = Ydata + pad_t() * padded_width + pad_l();
                math::CopyMatrix<CPUContext>(
                    sizeof(float),
                    height,
                    width,
                    Xdata,
                    width,
                    Ystart,
                    padded_width,
                    &context_);

    // Fixup areas where we need to reflect
    #define X(ph, pw)                 \
      int h = ph - pad_t();           \
      int w = pw - pad_l();           \
      h = max(h, -h);                 \
      h = min(h, 2 * height - h - 2); \
      w = max(w, -w);                 \
      w = min(w, 2 * width - w - 2);  \
      Ydata[ph * padded_width + pw] = Xdata[h * width + w]

                // Top part
                for (int ph = 0; ph < pad_t(); ++ph) {
                  for (int pw = 0; pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }

                // Bottom part
                for (int ph = padded_height - pad_b(); ph < padded_height; ++ph) {
                  for (int pw = 0; pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }

                // Interior
                for (int ph = pad_t(); ph < padded_height - pad_b(); ++ph) {
                  // Left
                  for (int pw = 0; pw < pad_l(); ++pw) {
                    X(ph, pw);
                  }
                  // Right
                  for (int pw = padded_width - pad_r(); pw < padded_width; ++pw) {
                    X(ph, pw);
                  }
                }
    #undef X

                // Do offset.
                Xdata += height * width;
                Ydata += padded_height * padded_width;
              }
            }
          } else {
            for (int n = 0; n < X.dim32(0); ++n) {
              for (int c = 0; c < channels; ++c) {
                for (int ph = 0; ph < padded_height; ++ph) {
                  for (int pw = 0; pw < padded_width; ++pw) {
                    int h = ph - pad_t();
                    int w = pw - pad_l();
                    // max(h, -h) does reflection over 0
                    h = max(h, -h);
                    // min(h, 2 * height - h - 2) does reflection over height.
                    h = min(h, 2 * height - h - 2);
                    w = max(w, -w);
                    w = min(w, 2 * width - w - 2);
                    Ydata[ph * padded_width + pw] = Xdata[h * width + w];
                  }
                }
                // Do offset.
                Xdata += height * width;
                Ydata += padded_height * padded_width;
              }
            }
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  // Bounds to the right range.
                  int h = min(height - 1, max(ph - pad_t(), 0));
                  int w = min(width - 1, max(pw - pad_l(), 0));
                  Ydata[ph * padded_width + pw] = Xdata[h * width + w];
                }
              }
              // Do offset.
              Xdata += height * width;
              Ydata += padded_height * padded_width;
            }
          }
          break;
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto* Y = Output(0);
      int height = X.dim32(1);
      int width = X.dim32(2);
      int channels = X.dim32(3);
      ConvPoolOpBase::SetOutputSize(X, Y, channels);
      const float* Xdata = X.data<float>();
      float* Ydata = Y->template mutable_data<float>();

      // The main loop
      int padded_height = Y->dim32(1);
      int padded_width = Y->dim32(2);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                int h = ph - pad_t();
                int w = pw - pad_l();
                const int pad_index = (ph * padded_width + pw) * channels;
                if (h < 0 || w < 0 || h >= height || w >= width) {
                  for (int c = 0; c < channels; ++c) {
                    Ydata[pad_index + c] = value_;
                  }
                } else {
                  const int input_index = (h * width + w) * channels;
                  for (int c = 0; c < channels; ++c) {
                    Ydata[pad_index + c] = Xdata[input_index + c];
                  }
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                int h = ph - pad_t();
                int w = pw - pad_l();
                // max(h, -h) does reflection over 0
                h = max(h, -h);
                // min(h, 2 * height - h - 2) does reflection over height.
                h = min(h, 2 * height - h - 2);
                w = max(w, -w);
                w = min(w, 2 * width - w - 2);
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  Ydata[pad_index + c] = Xdata[input_index + c];
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < X.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                int h = min(height - 1, max(ph - pad_t(), 0));
                int w = min(width - 1, max(pw - pad_l(), 0));
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  Ydata[pad_index + c] = Xdata[input_index + c];
                }
              }
            }
            // Do offset.
            Xdata += X.numel() / X.dim32(0);
            Ydata += Y->numel() / Y->dim32(0);
          }
          break;
      }
      return true;
        */
    }
}

///-----------------------------------
pub struct PadImageGradientOp<T,Context> {

    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    mode: PadMode,

    /**
      | Input: dY
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{PadImageGradient, 1}

num_outputs!{PadImageGradient, 1}

impl<T,Context> PadImageGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...),
            mode_(StringToPadMode(this->template GetSingleArgument<string>("mode", "constant"))) 

        CAFFE_ENFORCE(
            legacy_pad_ == LegacyPadding::NOTSET,
            "Padding layer only supports explicit pad values.");
        CAFFE_ENFORCE(
            dilation_h() == 1 && dilation_w() == 1,
            "Pooling op does not support dilation right now.");
        // Pad op does not use kernel sizes, so we set it to 1 for computing the
        // output size.
        kernel_.assign(pads_.size() / 2, 1);
        */
    }
}

impl PadImageGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(
          0,
          {dY.dim32(0),
           dY.dim32(1),
           dY.dim32(2) - pad_t() - pad_b(),
           dY.dim32(3) - pad_l() - pad_r()},
          at::dtype<float>());
      int padded_height = dY.dim32(2);
      int padded_width = dY.dim32(3);
      int channels = dX->dim32(1);
      int height = dX->dim32(2);
      int width = dX->dim32(3);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      math::Set<float, CPUContext>(dX->numel(), 0, dXdata, &context_);
      // The main loop
      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  if (!(h < 0 || w < 0 || h >= height || w >= width)) {
                    dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                  }
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = ph - pad_t();
                  int w = pw - pad_l();
                  // max(h, -h) does reflection over 0
                  h = max(h, -h);
                  // min(h, 2 * height - h - 2) does reflection over height.
                  h = min(h, 2 * height - h - 2);
                  w = max(w, -w);
                  w = min(w, 2 * width - w - 2);
                  dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int c = 0; c < channels; ++c) {
              for (int ph = 0; ph < padded_height; ++ph) {
                for (int pw = 0; pw < padded_width; ++pw) {
                  int h = min(height - 1, max(ph - pad_t(), 0));
                  int w = min(width - 1, max(pw - pad_l(), 0));
                  dXdata[h * width + w] += dYdata[ph * padded_width + pw];
                }
              }
              // Do offset.
              dXdata += height * width;
              dYdata += padded_height * padded_width;
            }
          }
          break;
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto* dX = Output(
          0,
          {dY.dim32(0),
           dY.dim32(1) - pad_t() - pad_b(),
           dY.dim32(2) - pad_l() - pad_r(),
           dY.dim32(3)},
          at::dtype<float>());
      int padded_height = dY.dim32(1);
      int padded_width = dY.dim32(2);
      int channels = dY.dim32(3);
      int height = dX->dim32(1);
      int width = dX->dim32(2);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      math::Set<float, CPUContext>(dX->numel(), 0, dXdata, &context_);

      switch (mode_) {
        case PadMode::CONSTANT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                int h = ph - pad_t();
                int w = pw - pad_l();
                const int pad_index = (ph * padded_width + pw) * channels;
                if (!(h < 0 || w < 0 || h >= height || w >= width)) {
                  const int input_index = (h * width + w) * channels;
                  for (int c = 0; c < channels; ++c) {
                    dXdata[input_index + c] += dYdata[pad_index + c];
                  }
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
        case PadMode::REFLECT:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                int h = ph - pad_t();
                int w = pw - pad_l();
                // max(h, -h) does reflection over 0
                h = max(h, -h);
                // min(h, 2 * height - h - 2) does reflection over height.
                h = min(h, 2 * height - h - 2);
                w = max(w, -w);
                w = min(w, 2 * width - w - 2);
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  dXdata[input_index + c] += dYdata[pad_index + c];
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
        case PadMode::EDGE:
          for (int n = 0; n < dY.dim32(0); ++n) {
            for (int ph = 0; ph < padded_height; ++ph) {
              for (int pw = 0; pw < padded_width; ++pw) {
                const int pad_index = (ph * padded_width + pw) * channels;
                // Bounds to the right range.
                int h = min(height - 1, max(ph - pad_t(), 0));
                int w = min(width - 1, max(pw - pad_l(), 0));
                const int input_index = (h * width + w) * channels;
                for (int c = 0; c < channels; ++c) {
                  dXdata[input_index + c] += dYdata[pad_index + c];
                }
              }
            }
            // Do offset.
            dXdata += dX->numel() / dX->dim32(0);
            dYdata += dY.numel() / dY.dim32(0);
          }
          break;
      }
      return true;
        */
    }
}

#[inline] pub fn string_to_pad_mode(mode: &String) -> PadMode {
    
    todo!();
    /*
        if (mode == "constant") {
        return PadMode::CONSTANT;
      } else if (mode == "reflect") {
        return PadMode::REFLECT;
      } else if (mode == "edge") {
        return PadMode::EDGE;
      } else {
        CAFFE_THROW("Unknown padding mode: " + mode);
      }
    */
}

register_cpu_operator!{PadImage,                  PadImageOp<f32, CPUContext>}

register_cpu_gradient_operator!{PadImageGradient, PadImageGradientOp<f32, CPUContext>}

pub struct GetPadImageGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetPadImageGradient<'a> {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "PadImageGradient", "", vector<string>{GO(0)}, vector<string>{GI(0)});
        */
    }
}

register_gradient!{PadImage, GetPadImageGradient}
