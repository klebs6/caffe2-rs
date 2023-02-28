crate::ix!();

/**
  | Resizes the spatial dimensions of the
  | input using bilinear interpolation.
  | 
  | The `width_scale` and `height_scale`
  | arguments control the size of the output,
  | which is given by: output_width = floor(input_width
  | * width_scale) output_height = floor(output_height
  | * height_scale)
  |
  */
pub struct UpsampleBilinearOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:      OperatorStorage,
    context:      Context,

    width_scale:  T,
    height_scale: T,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

num_inputs!{UpsampleBilinear, (1,2)}

num_outputs!{UpsampleBilinear, 1}

inputs!{UpsampleBilinear, 
    0 => ("X",      "Input tensor"),
    1 => ("scales", "1D, 2-element, Scales tensor, [height_scale, width_scale]")
}

outputs!{UpsampleBilinear, 
    0 => ("Y", "Output tensor")
}

args!{UpsampleBilinear, 
    0 => ("width_scale",  "Scale along width dimension"),
    1 => ("height_scale", "Scale along height dimension")
}

impl<T, Context> UpsampleBilinearOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            width_scale_(1),
            height_scale_(1) 

        if (HasArgument("width_scale")) {
          width_scale_ = static_cast<T>(
              this->template GetSingleArgument<float>("width_scale", 1));
        }
        if (HasArgument("height_scale")) {
          height_scale_ = static_cast<T>(
              this->template GetSingleArgument<float>("height_scale", 1));
        }
        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        */
    }
}

impl UpsampleBilinearOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

      if (InputSize() == 2) {
        const auto& scales = Input(1);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      const int batch_size = X.dim32(0);
      const int num_channels = X.dim32(1);
      const int input_height = X.dim32(2);
      const int input_width = X.dim32(3);
      int output_width = input_width * width_scale_;
      int output_height = input_height * height_scale_;
      auto* Y = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());

      const float* input = X.data<float>();
      float* output = Y->mutable_data<float>();
      int channels = num_channels * batch_size;

      const float rheight = (output_height > 1)
          ? (float)(input_height - 1) / (output_height - 1)
          : 0.f;
      const float rwidth =
          (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
      for (int h2 = 0; h2 < output_height; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < output_width; ++w2) {
          const float w1r = rwidth * w2;
          const int w1 = w1r;
          const int w1p = (w1 < input_width - 1) ? 1 : 0;
          const float w1lambda = w1r - w1;
          const float w0lambda = (float)1. - w1lambda;
          const float* Xdata = &input[h1 * input_width + w1];
          float* Ydata = &output[h2 * output_width + w2];
          for (int c = 0; c < channels; ++c) {
            Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
                h1lambda *
                    (w0lambda * Xdata[h1p * input_width] +
                     w1lambda * Xdata[h1p * input_width + w1p]);
            Xdata += input_width * input_height;
            Ydata += output_width * output_height;
          }
        }
      }

      return true;
        */
    }
}

///--------------------------------------

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UpsampleBilinearGradientOp<T, Context> {

    storage:      OperatorStorage,
    context:      Context,
    width_scale:  T,
    height_scale: T,

    /*
      | Input: dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{UpsampleBilinearGradient, (2,3)}

num_outputs!{UpsampleBilinearGradient, 1}

args!{UpsampleBilinearGradient, 
    0 => ("width_scale",  "Scale along width dimension"),
    1 => ("height_scale", "Scale along height dimension")
}

impl<T,Context> UpsampleBilinearGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            width_scale_(1),
            height_scale_(1) 

        width_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("width_scale", 1));
        height_scale_ = static_cast<T>(
            this->template GetSingleArgument<float>("height_scale", 1));
        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        */
    }
}

impl UpsampleBilinearGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
      const auto& dY = Input(0);
      const auto& X  = Input(1);

      if (InputSize() == 3) {
        const auto& scales = Input(2);
        CAFFE_ENFORCE_EQ(scales.dim(), 1);
        CAFFE_ENFORCE_EQ(scales.numel(), 2);
        const float* scales_data = scales.data<float>();
        height_scale_ = scales_data[0];
        width_scale_ = scales_data[1];
      }

      const auto inputDims = dY.sizes();
      CAFFE_ENFORCE_EQ(4, inputDims.size());
      const int batch_size = dY.dim32(0);
      const int num_channels = dY.dim32(1);
      const int input_height = dY.dim32(2);
      const int input_width = dY.dim32(3);
      const int output_height = X.dim32(2);
      const int output_width = X.dim32(3);
      auto* dX = Output(
          0,
          {batch_size, num_channels, output_height, output_width},
          at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.0f, dX->mutable_data<float>(), &context_);

      const float* dYdata = dY.data<float>();
      float* dXdata = dX->mutable_data<float>();
      int channels = num_channels * batch_size;

      const float rheight = (input_height > 1)
          ? (float)(output_height - 1) / (input_height - 1)
          : 0.f;
      const float rwidth =
          (input_width > 1) ? (float)(output_width - 1) / (input_width - 1) : 0.f;

      for (int h2 = 0; h2 < input_height; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < output_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (int w2 = 0; w2 < input_width; ++w2) {
          const float w1r = rwidth * w2;
          const int w1 = w1r;
          const int w1p = (w1 < output_width - 1) ? 1 : 0;
          const float w1lambda = w1r - w1;
          const float w0lambda = (float)1. - w1lambda;
          float* pos1 = &dXdata[h1 * output_width + w1];
          const float* pos2 = &dYdata[h2 * input_width + w2];
          for (int c = 0; c < channels; ++c) {
            pos1[0] += h0lambda * w0lambda * pos2[0];
            pos1[w1p] += h0lambda * w1lambda * pos2[0];
            pos1[h1p * output_width] += h1lambda * w0lambda * pos2[0];
            pos1[h1p * output_width + w1p] += h1lambda * w1lambda * pos2[0];
            pos1 += output_width * output_height;
            pos2 += input_width * input_height;
          }
        }
      }

      return true;
        */
    }
}

register_cpu_operator!{UpsampleBilinear,         UpsampleBilinearOp<f32, CPUContext>}

register_cpu_operator!{UpsampleBilinearGradient, UpsampleBilinearGradientOp<f32, CPUContext>}

pub struct GetUpsampleBilinearGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetUpsampleBilinearGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (def_.input().size() == 2) {
          // this is a hack to support the second input as dynamic
          // width_scale and height_scale to align with onnx change
          return SingleGradientDef(
              "UpsampleBilinearGradient",
              "",
              vector<string>{GO(0), I(0), I(1)},
              vector<string>{GI(0)});
        }
        return SingleGradientDef(
            "UpsampleBilinearGradient",
            "",
            vector<string>{GO(0), I(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{UpsampleBilinear, GetUpsampleBilinearGradient}
