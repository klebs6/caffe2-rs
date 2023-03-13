crate::ix!();

declare_export_caffe2_op_to_c10!{ResizeNearest3D}

/**
  | Resizes the spatial dimensions of the
  | input tensor using nearest neighbor
  | interpolation. The `width_scale`
  | and `height_scale` arguments control
  | the size of the output, which is given
  | by: output_width = floor(input_width
  | * width_scale) output_height = floor(output_height
  | * height_scale)
  | 
  | Assumptions:
  | 
  | - Only resize height and width
  | 
  | - Both width_scale and height_scale
  | scale are 2
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ResizeNearest3DOp<T,Context> {
    storage:        OperatorStorage,
    context:        Context,
    temporal_scale: T,
    height_scale:   T,
    width_scale:    T,
    order:          StorageOrder,

    /*
      | Input: X,
      | 
      | output: Y
      |
      */
}

num_inputs!{ResizeNearest3D, 1}

num_outputs!{ResizeNearest3D, 1}

inputs!{ResizeNearest3D, 
    0 => ("X", "Input tensor")
}

outputs!{ResizeNearest3D, 
    0 => ("Y", "Output tensor")
}

args!{ResizeNearest3D, 
    0 => ("temporal_scale", "Scale along temporal dimension"),
    1 => ("width_scale",    "Scale along width dimension"),
    2 => ("height_scale",   "Scale along height dimension")
}

impl<T,Context> ResizeNearest3DOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          temporal_scale_(1),
          height_scale_(1),
          width_scale_(1),
          order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))) 

        if (HasArgument("temporal_scale")) {
         temporal_scale_ = static_cast<T>(
             this->template GetSingleArgument<float>("temporal_scale", 1));
        }
        if (HasArgument("height_scale")) {
         height_scale_ = static_cast<T>(
             this->template GetSingleArgument<float>("height_scale", 1));
        }
        if (HasArgument("width_scale")) {
         width_scale_ = static_cast<T>(
             this->template GetSingleArgument<float>("width_scale", 1));
        }

        CAFFE_ENFORCE_GT(temporal_scale_, 0);
        CAFFE_ENFORCE_GT(height_scale_, 0);
        CAFFE_ENFORCE_GT(width_scale_, 0);

        CAFFE_ENFORCE(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
        */
    }
}
