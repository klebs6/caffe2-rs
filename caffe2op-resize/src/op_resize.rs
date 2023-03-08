crate::ix!();

declare_export_caffe2_op_to_c10!{ResizeNearest}

/**
  | Resizes the spatial dimensions of the
  | input using nearest neighbor interpolation.
  | The `width_scale` and `height_scale`
  | arguments control the size of the output,
  | which is given by:
  | 
  | -output_width = floor(input_width
  | * width_scale)
  | 
  | -output_height = floor(output_height
  | * height_scale)
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ResizeNearestOp<T,Context> {
    storage:       OperatorStorage,
    context:       Context,
    width_scale:   T,
    height_scale:  T,
    order:         StorageOrder,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

num_inputs!{ResizeNearest, (1,2)}

num_outputs!{ResizeNearest, 1}

inputs!{ResizeNearest, 
    0 => ("X", "Input tensor"),
    1 => ("scales", "1D, 2-element, Scales tensor, [height_scale, width_scale] -- the hack to support onnx spec")
}

outputs!{ResizeNearest, 
    0 => ("Y", "Output tensor")
}

args!{ResizeNearest, 
    0 => ("width_scale", "Scale along width dimension"),
    1 => ("height_scale", "Scale along height dimension")
}

inherit_onnx_schema!{ResizeNearest, "Upsample"}

impl<T,Context> ResizeNearestOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            width_scale_(1),
            height_scale_(1),
            order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))) 

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

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
}
