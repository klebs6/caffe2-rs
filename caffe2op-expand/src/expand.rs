crate::ix!();

/**
  | Broadcast the input tensor to a materialized
  | new tensor using given shape.
  | 
  | Broadcast rule is similar to "numpy.array(input)
  | numpy.ones(shape)":
  | 
  | Dimensions are right alignment;
  | 
  | Two corresponding dimensions must
  | have the same value, or one of them equals
  | to 1.
  | 
  | In order to align with PyTorch's `expand`,
  | `shape` is allowed to have entries equal
  | to -1, which means to preserve the size
  | of the corresponding dimension in `X`
  | (so it's actually equivalent to equal
  | to 1).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ExpandOp<InputTypes, Context> {
    storage:   OperatorStorage,
    context:   Context,
    phantomIT: PhantomData<InputTypes>,
}

num_inputs!{Expand, 2}

num_outputs!{Expand, 1}

inputs!{Expand, 
    0 => ("X", "(*Tensor`<NumericType>`*): input tensor"),
    1 => ("shape", "(*Tensor`<int>`*): expand shape")
}

outputs!{Expand, 
    0 => ("Y", "(*Tensor`<NumericType>`*): expanded tensor")
}

register_cpu_operator!{
    Expand,
    ExpandOp<
        TensorTypes<i32, i64, f32, f64>,
        CPUContext>
}

register_cuda_operator!{
    Expand,
    ExpandOp<
        TensorTypes<i32, i64, f32, f64>,
        CUDAContext>
}

impl<InputTypes,Context> ExpandOp<InputTypes,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const auto& Y_shape_tensor = Input(1);
            std::vector<int64_t> shape_dims(Y_shape_tensor.numel());
            context_.template CopyToCPU<int64_t>(
                Y_shape_tensor.numel(),
                Y_shape_tensor.template data<int64_t>(),
                shape_dims.data());

            const int ndim = shape_dims.size();
            const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
            std::vector<int> Y_dims;
            Y_dims.reserve(std::max(ndim, X.dim()));
            // ndim, X.ndim() might equal to 0
            for (int i = ndim - 1, j = X.dim() - 1; i >= 0 || j >= 0; --i, --j) {
              const int shape_x = (j >= 0 ? X_dims[j] : 1);
              // In PyTorch expand treats -1 as a special value to indicate
              // preserving the size of that dimension.
              const int shape_y = ((i >= 0 && shape_dims[i] > 0) ? shape_dims[i] : 1);

              CAFFE_ENFORCE(
                  shape_x == 1 || shape_y == 1 || shape_x == shape_y,
                  "Dimensions format invalid.");
              Y_dims.push_back(std::max(shape_x, shape_y));
            }
            std::reverse(Y_dims.begin(), Y_dims.end());
            // TODO: remove when the function in math are changed to use vector<int64_t>
            std::vector<int64_t> Y_dims_int64;
            std::copy(Y_dims.begin(), Y_dims.end(), std::back_inserter(Y_dims_int64));
            auto* Y = Output(0, Y_dims_int64, at::dtype<T>());
            math::Broadcast<T, Context>(
                X_dims.size(),
                X_dims.data(),
                Y_dims.size(),
                Y_dims.data(),
                T(1),
                X.template data<T>(),
                Y->template mutable_data<T>(),
                &context_);
            return true;
        */
    }
}
