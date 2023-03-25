crate::ix!();

/**
  | Carries out ROI Pooling for Faster-RCNN.
  | 
  | Depending on the mode, there are multiple
  | output cases:
  | 
  | Output case #1: Y, argmaxes (train mode)
  | 
  | Output case #2: Y (test mode)
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIPoolOp<T,Context> {

    storage:        OperatorStorage,
    context:        Context,
    is_test:        bool,
    order:          StorageOrder,
    pooled_height:  i32,
    pooled_width:   i32,
    spatial_scale:  f32,

    /**
      | Input: X, rois
      |
      | Output case #1: Y, argmaxes (train mode)
      |
      | Output case #2: Y           (test mode)
      */
    phantom: PhantomData<T>,
}

num_inputs!{RoIPool, 2}

num_outputs!{RoIPool, (1,2)}

inputs!{RoIPool, 
    0 => ("X",    "The input 4-D tensor of data. Only NCHW order is currently supported."),
    1 => ("rois", "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].")
}

outputs!{RoIPool, 
    0 => ("Y",        "RoI pooled output 4-D tensor of shape (num_rois, channels, pooled_h, pooled_w)."),
    1 => ("argmaxes", "Argmaxes corresponding to indices in X used for gradient computation. Only output if arg is_test is false.")
}

args!{RoIPool, 
    0 => ("is_test",        "If set, run in test mode and skip computation of argmaxes (used for gradient computation). Only one output tensor is produced. (Default: false)."),
    1 => ("order",          "A StorageOrder string (Default: NCHW)."),
    2 => ("pooled_h",       "The pooled output height (Default: 1)."),
    3 => ("pooled_w",       "The pooled output width (Default: 1)."),
    4 => ("spatial_scale",  "Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling (Default: 1.0).")
}

tensor_inference_function!{RoIPool, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const StorageOrder order = StringToStorageOrder(
          helper.GetSingleArgument<string>("order", "NCHW"));
      const TensorShape& X = in[0];
      const int num_channels =
          (order == StorageOrder::NCHW ? X.dims(1) : X.dims(3));
      const TensorShape& R = in[1];
      const int num_rois = R.dims(0);
      const int pooled_height = helper.GetSingleArgument<int>("pooled_h", 1);
      const int pooled_width = helper.GetSingleArgument<int>("pooled_w", 1);
      TensorShape Y = CreateTensorShape(
          vector<int>({num_rois, num_channels, pooled_height, pooled_width}),
          X.data_type());

      bool is_test = helper.GetSingleArgument<int>(OpSchema::Arg_IsTest, 0);
      if (!is_test) {
        TensorShape argmaxes = Y;
        argmaxes.set_data_type(TensorProto_DataType_INT32);
        return vector<TensorShape>({Y, argmaxes});
      } else {
        return vector<TensorShape>({Y});
      }
    }) */
}

impl<T,Context> RoIPoolOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            is_test_( this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)) 

        CAFFE_ENFORCE(
            (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 2),
            "Output size mismatch.");
        CAFFE_ENFORCE_GT(spatial_scale_, 0);
        CAFFE_ENFORCE_GT(pooled_height_, 0);
        CAFFE_ENFORCE_GT(pooled_width_, 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
}
