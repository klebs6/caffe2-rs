crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    CPUContext,
    StorageOrder,
    GradientMakerBase,
};

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

impl RoIPoolOp<f32,CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0); // Input data to pool
      const auto& R = Input(1); // RoIs
      auto* Y = Output(0); // RoI pooled data
      auto* A = is_test_ ? nullptr : Output(1); // argmaxes

      // Each ROI is of the form [batch_index x1 y1 x2 y2]
      CAFFE_ENFORCE_EQ(R.dim32(1), 5);

      // TODO: Handle the storage_order properly to get the NCWH.
      int batch_size = X.dim32(0);
      int channels = X.dim32(1);
      int height = X.dim32(2);
      int width = X.dim32(3);
      int num_rois = R.dim32(0);

      Y->Resize(num_rois, channels, pooled_height_, pooled_width_);
      if (!is_test_) {
        A->Resize(Y->sizes());
      }

      const float* Xdata = X.data<float>();
      const float* rois = R.data<float>();
      float* Ydata = Y->template mutable_data<float>();
      int* argmax_data = is_test_ ? nullptr : A->template mutable_data<int>();

      // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
      for (int n = 0; n < num_rois; ++n) {
        int roi_batch_id = rois[0];
        int roi_start_w = round(rois[1] * spatial_scale_);
        int roi_start_h = round(rois[2] * spatial_scale_);
        int roi_end_w = round(rois[3] * spatial_scale_);
        int roi_end_h = round(rois[4] * spatial_scale_);
        CAFFE_ENFORCE_GE(roi_batch_id, 0);
        CAFFE_ENFORCE_LT(roi_batch_id, batch_size);

        // Force malformed ROIs to be 1x1
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);

        const float bin_size_h =
            static_cast<float>(roi_height) / static_cast<float>(pooled_height_);
        const float bin_size_w =
            static_cast<float>(roi_width) / static_cast<float>(pooled_width_);

        const float* batch_data = Xdata + roi_batch_id * X.size_from_dim(1);

        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              // Compute pooling region for this output unit:
              //  start (included) = floor(ph * roi_height / pooled_height_)
              //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
              int hstart =
                  static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
              int wstart =
                  static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
              int hend =
                  static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
              int wend =
                  static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

              // Add roi offsets and clip to input boundaries
              hstart = min(max(hstart + roi_start_h, 0), height);
              hend = min(max(hend + roi_start_h, 0), height);
              wstart = min(max(wstart + roi_start_w, 0), width);
              wend = min(max(wend + roi_start_w, 0), width);

              const int pool_index = ph * pooled_width_ + pw;

              // Define an empty pooling region to be zero
              bool is_empty = (hend <= hstart) || (wend <= wstart);
              Ydata[pool_index] = is_empty ? 0 : -FLT_MAX;
              if (!is_test_) {
                // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                argmax_data[pool_index] = -1;
              }

              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * width + w;
                  if (batch_data[index] > Ydata[pool_index]) {
                    Ydata[pool_index] = batch_data[index];
                    if (!is_test_) {
                      argmax_data[pool_index] = index;
                    }
                  }
                }
              }
            }
          }
          // Increment all data pointers by one channel
          batch_data += X.size_from_dim(2);
          Ydata += Y->size_from_dim(2);
          if (!is_test_) {
            argmax_data += A->size_from_dim(2);
          }
        }
        // Increment ROI data pointer
        rois += R.size_from_dim(1);
      }

      return true;
        */
    }
}

///-----------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RoIPoolGradientOp<T,Context> {

    storage:        OperatorStorage,
    context:        Context,

    spatial_scale:  f32,
    pooled_height:  i32,
    pooled_width:   i32,
    order:          StorageOrder,

    /**
      | Input: X, rois, argmaxes, dY (aka
      | "gradOutput")
      |
      | Output: dX (aka "gradInput")
      */
    phantom:        PhantomData<T>,
}

num_inputs!{RoIPoolGradient, 4}

num_outputs!{RoIPoolGradient, 1}

impl<T,Context> RoIPoolGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            spatial_scale_( this->template GetSingleArgument<float>("spatial_scale", 1.)),
            pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
            pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
            order_(StringToStorageOrder( this->template GetSingleArgument<string>("order", "NCHW"))) 

        CAFFE_ENFORCE_GT(spatial_scale_, 0);
        CAFFE_ENFORCE_GT(pooled_height_, 0);
        CAFFE_ENFORCE_GT(pooled_width_, 0);
        CAFFE_ENFORCE_EQ(
            order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }
}

register_cpu_operator!{RoIPool, RoIPoolOp<float, CPUContext>}

register_cpu_operator!{RoIPoolGradient, RoIPoolGradientOp<float, CPUContext>}

pub struct GetRoIPoolGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRoIPoolGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RoIPoolGradient",
            "",
            vector<string>{I(0), I(1), O(1), GO(0)},
            vector<string>{GI(0)});
        */
    }
}

register_gradient!{RoIPool, GetRoIPoolGradient}
