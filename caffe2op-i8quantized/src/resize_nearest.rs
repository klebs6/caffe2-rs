crate::ix!();

/**
  | Resizes the spatial dimensions of the
  | input using nearest neighbor interpolation.
  | 
  | The `width_scale` and `height_scale`
  | arguments control the size of the output,
  | which is given by:
  | 
  | output_width = floor(input_width
  | * width_scale)
  | 
  | output_height = floor(output_height
  | height_scale)
  |
  */
pub struct Int8ResizeNearestOp {
    storage:      OperatorStorage,
    context:      CPUContext,

    width_scale:  f32,
    height_scale: f32,
    output_dims:  Vec<i32>,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

register_cpu_operator!{Int8ResizeNearest, int8::Int8ResizeNearestOp}

num_inputs!{Int8ResizeNearest, 1}

num_outputs!{Int8ResizeNearest, 1}

inputs!{Int8ResizeNearest, 
    0 => ("X", "Input Int8 tensor")
}

outputs!{Int8ResizeNearest, 
    0 => ("Y", "Output Int8 tensor")
}

args!{Int8ResizeNearest, 
    0 => ("Y_scale",      "Output tensor quantization scale"),
    1 => ("Y_zero_point", "Output tensor quantization offset"),
    2 => ("width_scale",  "Scale along width dimension"),
    3 => ("height_scale", "Scale along height dimension"),
    4 => ("output_size",  "Output dimensions (HxW). If specified this takes precedence over scale values.")
}

impl Int8ResizeNearestOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...) 

        width_scale_ = this->template GetSingleArgument<float>("width_scale", 1);
        height_scale_ = this->template GetSingleArgument<float>("height_scale", 1);
        output_dims =
            this->template GetRepeatedArgument<int>("output_size", vector<int>{});
        CAFFE_ENFORCE_GT(width_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Assume NHWC layout.
        const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();

        CAFFE_ENFORCE_EQ(4, X.t.dim());

        const int N = X.t.dim32(0);
        const int IH = X.t.dim32(1);
        const int IW = X.t.dim32(2);
        const int C = X.t.dim32(3);
        if (!output_dims.empty()) {
          CAFFE_ENFORCE_EQ(
              2, output_dims.size(), "Int8ResizeNearest expects 2 dim output size");
          height_scale_ = output_dims[0] / IH;
          width_scale_ = output_dims[1] / IW;
        }
        const int OW = IW * width_scale_;
        const int OH = IH * height_scale_;
        ReinitializeTensor(&Y->t, {N, OH, OW, C}, at::dtype<uint8_t>().device(CPU));
        Y->scale = X.scale;
        Y->zero_point = X.zero_point;

        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);

        const uint8_t* Xdata = X.t.data<uint8_t>();
        uint8_t* Ydata = Y->t.mutable_data<uint8_t>();

        for (int n = 0; n < N; ++n) {
          for (int y = 0; y < OH; ++y) {
            const int in_y = std::min((int)(y / height_scale_), (IH - 1));
            for (int x = 0; x < OW; ++x) {
              const int in_x = std::min((int)(x / width_scale_), (IW - 1));
              std::memcpy(
                  &Ydata[C * x + C * OW * y + C * OW * OH * n],
                  &Xdata[C * in_x + C * IW * in_y + C * IW * IH * n],
                  C);
            }
          }
        }
        return true;
        */
    }
}

