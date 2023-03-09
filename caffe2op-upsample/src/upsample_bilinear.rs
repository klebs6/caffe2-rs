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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UpsampleBilinearOp<T, Context> {

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
