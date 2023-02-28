crate::ix!();

use crate::{
    OperatorStorage,
    Tensor,
    CPUContext
};

/**
  | PiecewiseLinearTransform takes inputs
  | -- predictions, a 2-D or 1-D tensor (Tensor)
  | of size (batch_size x prediction_dimensions).
  | 
  | The piecewise linear functions are
  | stored in bounds, slopes and intercepts.
  | The output tensor has the same shape
  | of input `predictions` and contains
  | the predictions transformed by the
  | piecewise linear functions.
  | 
  | Each column of predictions has its own
  | piecewise linear transformation functions.
  | 
  | Therefore the size of piecewise function
  | parameters are pieces x prediction_dimensions,
  | except for binary predictions where
  | only the positive prediction needs
  | them.
  | 
  | -----------
  | @note
  | 
  | in each piece, low bound is excluded
  | while high bound is included. Also the
  | piecewise linear function must be continuous.
  | 
  | Notes
  | 
  | - If the input is binary predictions
  | (Nx2 or Nx1 tensor), set the binary arg
  | to true so that one group of piecewise
  | linear functions is needed (see details
  | below).
  | 
  | - The transform parameters (bounds,
  | slopes, intercepts) can be passed either
  | through args or through input blobs.
  | 
  | - If we have multiple groups of piecewise
  | linear functions, each group has the
  | same number of pieces.
  | 
  | - If a prediction is out of the bounds,
  | it is capped to the smallest or largest
  | bound.
  |
  */
pub struct PiecewiseLinearTransformOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:             OperatorStorage,
    context:             Context,

    binary:              bool,
    bounds_from_arg:     Vec<T>,
    slopes_from_arg:     Vec<T>,
    intercepts_from_arg: Vec<T>,
    bounds_device:       Tensor, //{Context::GetDeviceType()};
    intercepts_device:   Tensor, //{Context::GetDeviceType()};
    slopes_device:       Tensor, //{Context::GetDeviceType()};
    gpu_copied:          bool,   // = false;

    /**
      | If true, the piecewise linear functions
      | are passed through args, otherwise,
      | they are passed through Input blobs.
      |
      */
    transform_param_from_arg: bool,
}

pub type PiecewiseLinearTransformOpFloatCPU = PiecewiseLinearTransformOp<f32, CPUContext>;

num_inputs!{PiecewiseLinearTransform, (1,4)}

num_outputs!{PiecewiseLinearTransform, 1}

inputs!{PiecewiseLinearTransform, 
    0 => ("predictions",           "2-D tensor (Tensor) of size (num_batches x num_classes) containing scores"),
    1 => ("bounds (optional)",     "See bounds in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    2 => ("slopes (optional)",     "See slopes in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    3 => ("intercepts (optional)", "See intercepts in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.")
}

outputs!{PiecewiseLinearTransform, 
    0 => ("transforms",            "2-D tensor (Tensor) of size (num_batches x num_classes) containing transformed predictions")
}

args!{PiecewiseLinearTransform, 
    0 => ("bounds",                "1-D vector of size (prediction_dimensions x (pieces+1)) contain the upper bounds of each piece of linear function. One special case is the first bound is the lower bound of whole piecewise function and we treat it the same as the left most functions. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    1 => ("slopes",                "1-D vector of size (prediction_dimensions x pieces) containing the slopes of linear function"),
    2 => ("intercepts",            "1-D vector of size (prediction_dimensions x pieces) containing the intercepts of linear function"),
    3 => ("binary",                "If set true, we assume the input is a Nx1 or Nx2 tensor. If it is Nx1 tensor, it is positive predictions. If the input is Nx2 tensor, its first column is negative predictions and second column is positive and negative + positive = 1. We just need one group of piecewise linear functions for the positive predictions.")
}

input_tags!{
    PiecewiseLinearTransformOp {
        Predictions,
        Bounds,
        Slopes,
        Intercepts
    }
}

impl<T, Context> PiecewiseLinearTransformOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        binary_ = this->template GetSingleArgument<bool>("binary", false);

        // Retrieve transform params (i.e., the linear functions).
        bounds_from_arg_ = this->template GetRepeatedArgument<T>("bounds");
        slopes_from_arg_ = this->template GetRepeatedArgument<T>("slopes");
        intercepts_from_arg_ = this->template GetRepeatedArgument<T>("intercepts");
        transform_param_from_arg_ = CheckTransParamFromArg();
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return binary_ ? TransformBinary() : TransformGeneral();
        */
    }

    /**
      | num_func_per_group is the number of pieces
      | of linear functions of each group.
      |
      | num_group: The number of groups of linear
      | functions. Each group is for transforming
      | one column of predictions.
      */
    #[inline] pub fn infer_num_functions_per_group(
        &mut self, 
        num_bounds:          i64,
        num_slopes:          i64,
        num_intercepts:      i64,
        num_func_per_group:  *mut i64,
        num_group:           *mut i64)  
    {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(num_slopes, num_intercepts);

        // This is based on the facts:
        // 1. in each group, the num of bounds minus the num of slopes is 1;
        // 2. each group has the same number of pieces.
        *num_group = num_bounds - num_slopes;
        CAFFE_ENFORCE_GT(*num_group, 0);
        if (binary_) {
          CAFFE_ENFORCE_EQ(*num_group, 1);
        }
        *num_func_per_group = num_slopes / *num_group;
        CAFFE_ENFORCE_GT(*num_func_per_group, 0);
        CAFFE_ENFORCE_EQ(num_slopes % *num_group, 0);
        */
    }
    
    #[inline] pub fn check_bounds_sorted(
        &mut self, 
        bounds:                *const T,
        num_bounds_per_group:  i64,
        num_group:             i64) -> bool 
    {
        
        todo!();
        /*
            const T* start = bounds;
        for (int64_t i = 0; i < num_group; i++) {
          if (!std::is_sorted(start, start + num_bounds_per_group)) {
            return false;
          }
          start += num_bounds_per_group;
        }
        return true;
        */
    }

    /**
      | Returns true if the transform params from
      | arg are valid.
      |
      | Otherwise, we will assume the transform
      | params will pass from Input blobs.
      */
    #[inline] pub fn check_trans_param_from_arg(&mut self) -> bool {
        
        todo!();
        /*
            int good_param = 0;
        good_param += bounds_from_arg_.size() > 0;
        good_param += slopes_from_arg_.size() > 0;
        good_param += intercepts_from_arg_.size() > 0;
        CAFFE_ENFORCE(
            good_param == 0 || good_param == 3,
            "bounds, slopes, intercepts must be all set or all not set");
        if (good_param == 3) {
          int64_t num_func_per_group;
          int64_t num_group;
          InferNumFunctionsPerGroup(
              bounds_from_arg_.size(),
              slopes_from_arg_.size(),
              intercepts_from_arg_.size(),
              &num_func_per_group,
              &num_group);
          CAFFE_ENFORCE(
              CheckBoundsSorted(
                  bounds_from_arg_.data(), num_func_per_group + 1, num_group),
              "bounds must be sorted for each group");
        }

        return good_param == 3;
        */
    }

    #[inline] pub fn get_trans_param_data(&mut self, 
        bounds:              *const *const T,
        slopes:              *const *const T,
        intercepts:          *const *const T,
        num_func_per_group:  *mut i64,
        num_group:           *mut i64)  
    {

        todo!();
        /*
            int64_t num_bounds;
        int64_t num_slopes;
        int64_t num_intercepts;

        if (transform_param_from_arg_) {
          CAFFE_ENFORCE_EQ(InputSize(), 1);
          *bounds = bounds_from_arg_.data();
          *slopes = slopes_from_arg_.data();
          *intercepts = intercepts_from_arg_.data();
          num_bounds = bounds_from_arg_.size();
          num_slopes = slopes_from_arg_.size();
          num_intercepts = intercepts_from_arg_.size();
        } else {
          CAFFE_ENFORCE_EQ(InputSize(), 4);
          auto& bounds_input = Input(BOUNDS);
          auto& slopes_input = Input(SLOPES);
          auto& intercepts_input = Input(INTERCEPTS);
          *bounds = bounds_input.template data<T>();
          *slopes = slopes_input.template data<T>();
          *intercepts = intercepts_input.template data<T>();
          num_bounds = bounds_input.numel();
          num_slopes = slopes_input.numel();
          num_intercepts = intercepts_input.numel();
        }
        InferNumFunctionsPerGroup(
            num_bounds, num_slopes, num_intercepts, num_func_per_group, num_group);
        */
    }
    
    #[inline] pub fn transform_general(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        CAFFE_ENFORCE_EQ(X.dim(), 2);
        int64_t N = X.dim32(0);
        int64_t M = X.dim32(1);
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        const auto* Xdata = X.template data<T>();
        T* Ydata = Y->template mutable_data<T>();

        const T* bounds;
        const T* slopes;
        const T* intercepts;
        int64_t num_func_per_group;
        int64_t num_group;
        GetTransParamData(
            &bounds, &slopes, &intercepts, &num_func_per_group, &num_group);
        CAFFE_ENFORCE_EQ(num_group, M);

        for (int64_t j = 0; j < M; ++j) {
          const T* bounds_group = bounds + j * (num_func_per_group + 1);
          const T* slopes_group = slopes + j * num_func_per_group;
          const T* intercepts_group = intercepts + j * num_func_per_group;
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i * M + j] = PiecewiseLinearTransform(
                Xdata[i * M + j],
                bounds_group,
                slopes_group,
                intercepts_group,
                num_func_per_group);
          }
        }
        return true;
        */
    }
    
    #[inline] pub fn transform_binary(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(PREDICTIONS);

        CAFFE_ENFORCE(X.dim() == 1 || X.dim() == 2);
        int64_t N = X.dim32(0);
        int64_t M = X.dim() == 2 ? X.dim32(1) : 1;
        CAFFE_ENFORCE(
            M == 1 || M == 2,
            "If binary is set to true, the input must be Nx2 or Nx1 tensor");
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        const auto* Xdata = X.template data<T>();
        T* Ydata = Y->template mutable_data<T>();

        const T* bounds;
        const T* slopes;
        const T* intercepts;
        int64_t num_func_per_group;
        int64_t num_group;
        GetTransParamData(
            &bounds, &slopes, &intercepts, &num_func_per_group, &num_group);
        CAFFE_ENFORCE_EQ(num_group, 1);

        if (M == 1) {
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i] = PiecewiseLinearTransform(
                Xdata[i], bounds, slopes, intercepts, num_func_per_group);
          }
        } else {
          for (int64_t i = 0; i < N; ++i) {
            Ydata[i * M + 1] = PiecewiseLinearTransform(
                Xdata[i * M + 1], bounds, slopes, intercepts, num_func_per_group);
            Ydata[i * M] = 1.0f - Ydata[i * M + 1];
          }
        }

        return true;
        */
    }
    
    #[inline] pub fn piecewise_linear_transform(
        &mut self, 
        x:                    T,
        bounds:               *const T,
        slopes:               *const T,
        intercepts:           *const T,
        num_func_per_group:   i64) -> T 
    {
        todo!();
        /*
            T y = 0;
        // deal with samples out of bounds
        // make it the same as the upper/lower bound value
        if (x <= bounds[0]) {
          y = slopes[0] * bounds[0] + intercepts[0];
        } else if (x >= bounds[num_func_per_group]) {
          y = slopes[num_func_per_group - 1] * bounds[num_func_per_group] +
              intercepts[num_func_per_group - 1];
        } else {
          auto low_bound =
              std::lower_bound(bounds, bounds + num_func_per_group + 1, x);
          int bounds_idx = low_bound - bounds - 1;
          // compute the piecewise linear transformation as Y
          y = slopes[bounds_idx] * x + intercepts[bounds_idx];
        }
        return y;
        */
    }
}

register_cpu_operator!{
    PiecewiseLinearTransform,
    PiecewiseLinearTransformOp<f32, CPUContext>
}

should_not_do_gradient!{PiecewiseLinearTransform}

