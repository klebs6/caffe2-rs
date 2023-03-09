crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TileGradientOp<Context> {

    storage: OperatorStorage,
    context: Context,
    tiles:   i32,
    axis:    i32,
    ones:    Tensor,
}

num_inputs!{TileGradient, (1,3)}

num_outputs!{TileGradient, 1}

register_cpu_operator!{TileGradient, TileGradientOp<CPUContext>}

impl<Context> TileGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(std::int32_t, "tiles", tiles_, 1),
            OP_SINGLE_ARG(std::int32_t, "axis", axis_, 0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<std::int32_t, std::int64_t, float, double>>::
            call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            if (InputSize() > 1) {
          // We potentially have tiles and/or axis specified as inputs
          // as well. We will check for them in that order. In other words:
          // InputSize() == 2: tiles is specified
          // InputSize() == 3: tiles is specified and axis.
          // Anything specified as input will override the arguments
          CAFFE_ENFORCE(
              Input(1).dim() == 1 && Input(1).numel() == 1,
              "Input `tiles` should be a vector of size 1.");
          tiles_ = GetArgFromTensor(Input(1));
          if (InputSize() > 2) {
            CAFFE_ENFORCE(
                Input(2).dim() == 1 && Input(2).numel() == 1,
                "Input `axis` should be a vector of size 1.");
            axis_ = GetArgFromTensor(Input(2));
          } else {
            CAFFE_ENFORCE(
                OperatorStorage::HasArgument("axis"),
                "Argument `axis` is missing and was not specified as input.");
          }
        } else {
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("tiles"),
              "Argument `tiles` is missing and was not specified as input.");
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("axis"),
              "Argument `axis` is missing and was not specified as input.");
        }

        const auto& dY = Input(0);
        auto* dX = Output(0);
        const int axis = dY.canonical_axis_index(axis_);

        // reshape output to be input "untiled" along the axis
        std::vector<std::int64_t> X_dims = dY.sizes().vec();
        CAFFE_ENFORCE_EQ(X_dims[axis] % tiles_, 0);
        X_dims[axis] /= tiles_;
        dX->Resize(X_dims);

        // size up to (and not including) axis
        const int outer_size = dX->size_to_dim(axis);
        // size from axis up
        const int inner_size = dX->size_from_dim(axis);

        /**
         * How this works:
         * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times along axis 1
         * (column).
         * This is equivalent to multiplying by a vector of 1s transposed.
         * The gradient of this is all 1s in the shape of the input matrix
         * (call it X).
         * So the output gradient should be the matrix multiplication result
         * of input gradient (gradient of tiled tensor output) and X.
         */
        const T* dY_data = dY.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        return DoTileGradient<T>(outer_size, inner_size, dY_data, dX_data);
        */
    }
    
    #[inline] pub fn get_arg_from_tensor(&mut self, tensor: &Tensor) -> i32 {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            tensor.IsType<std::int32_t>() || tensor.IsType<std::int64_t>());
        std::int32_t val = -1;
        if (tensor.IsType<std::int32_t>()) {
          context_.template CopyToCPU<std::int32_t>(
              1, tensor.data<std::int32_t>(), &val);
        } else if (tensor.IsType<std::int64_t>()) {
          std::int64_t val_int64;
          context_.template CopyToCPU<std::int64_t>(
              1, tensor.data<std::int64_t>(), &val_int64);
          val = static_cast<std::int32_t>(val_int64);
        }
        return val;
        */
    }
    
    #[inline] pub fn do_tile_gradient<T>(&mut self, 
        outer_size: i32,
        inner_size: i32,
        dy:         *const T,
        dx:         *mut T) -> bool {

        todo!();
        /*
            if (inner_size == 1) {
          const std::array<int, 2> dY_dims = {outer_size, tiles_};
          const std::array<int, 2> dX_dims = {outer_size, 1};
          math::ReduceSum<T, Context>(
              2, dY_dims.data(), dX_dims.data(), T(1), dY, dX, &context_);
        } else {
          math::CopyMatrix<T, Context>(
              outer_size,
              inner_size,
              dY,
              inner_size * tiles_,
              dX,
              inner_size,
              &context_);
          for (int i = 0; i < outer_size; ++i) {
            const T* dY_ptr = dY + i * tiles_ * inner_size;
            T* dX_ptr = dX + i * inner_size;
            for (int j = 1; j < tiles_; ++j) {
              math::Add<T, Context>(
                  inner_size, dX_ptr, dY_ptr + j * inner_size, dX_ptr, &context_);
            }
          }
        }
        return true;
        */
    }
}
