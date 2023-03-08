crate::ix!();

/**
  | Computes the mean and variance of the
  | input tensor's element along the provided
  | axes. The resulted tensor has the same
  | rank as the input if keepdims equals
  | True.
  | 
  | If keepdims equals False, then the resulted
  | tensor have the reduced dimension pruned.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MomentsOp<T,Context> {
    storage:   OperatorStorage,
    context:   Context,
    axes:      Vec<i32>,
    keep_dims: i32,
    phantom:   PhantomData<T>,
}

num_inputs!{Moments, 1}

num_outputs!{Moments, 2}

inputs!{Moments, 
    0 => ("data", "An input tensor.")
}

outputs!{Moments, 
    0 => ("mean", "Reduced mean tensor."),
    1 => ("variance", "Reduced variance tensor.")
}

args!{Moments, 
    0 => ("axes", "A list of integers, along which to reduce. If axes is not provided, the op computes the element-wise mean and variance."),
    1 => ("keepdims", "Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).")
}

impl<T,Context> MomentsOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes")),
            OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.begin(), axes_.end(), 0);
        } else {
          std::sort(axes_.begin(), axes_.end());
          CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
          CAFFE_ENFORCE_LT(
              axes_.back(),
              ndim,
              "Axes ids must be smaller than the dimensions of input.");
        }
        const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
        std::vector<int> Y_dims = X_dims;
        for (const int axis : axes_) {
          Y_dims[axis] = 1;
        }
        std::vector<std::int64_t> output_dims;
        output_dims.reserve(ndim);
        std::size_t cur_axis = 0;
        for (int i = 0; i < ndim; ++i) {
          if (cur_axis < axes_.size() && i == axes_[cur_axis]) {
            if (keep_dims_) {
              output_dims.push_back(1);
            }
            ++cur_axis;
          } else {
            output_dims.push_back(X_dims[i]);
          }
        }
        auto* mean = Output(0, output_dims, at::dtype<T>());
        auto* var = Output(1, output_dims, at::dtype<T>());
        math::Moments<float, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            X.template data<T>(),
            mean->template mutable_data<T>(),
            var->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

register_cpu_operator!{
    Moments,
    MomentsOp<f32, CPUContext>
}

register_cpu_operator!{
    MomentsGradient, 
    MomentsGradientOp<f32, CPUContext>
}
