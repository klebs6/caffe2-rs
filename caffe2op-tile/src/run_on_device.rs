crate::ix!();

impl<Context> TileOp<Context> {
    
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

          // Because of a bug in original code, temporarily adds this part to keep
          // backward compatibility.
          // TODO(yangxm): Remove this part when prod runtime upgraded with fixed
          // model config.
          if (Input(1).template IsType<std::int64_t>()) {
            axis_ = 0;
          }

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

        const auto& X = Input(0);
        auto* Y = Output(0);
        const int axis = X.canonical_axis_index(axis_);

        // reshape output to be input tiled along the axis
        std::vector<std::int64_t> Y_dims = X.sizes().vec();
        Y_dims[axis] *= tiles_;
        Y->Resize(Y_dims);

        // size up to (and not including) axis
        const int outer_size = X.size_to_dim(axis);
        // size from axis up
        const int inner_size = X.size_from_dim(axis);

        const T* X_data = X.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        return DoTile<T>(outer_size, inner_size, X_data, Y_data);
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
    
    #[inline] pub fn do_tile<T>(&mut self, 
        outer_size: i32,
        inner_size: i32,
        x:          *const T,
        y:          *mut T) -> bool {

        todo!();
        /*
            if (inner_size == 1) {
          EigenArrayMap<T> Y_arr(Y, tiles_, outer_size);
          for (int i = 0; i < outer_size; ++i) {
            Y_arr.col(i) = X[i];
          }
        } else {
          ConstEigenArrayMap<T> X_arr(X, inner_size, outer_size);
          for (int i = 0; i < outer_size; ++i) {
            EigenArrayMap<T>(Y + i * tiles_ * inner_size, inner_size, tiles_)
                .colwise() = X_arr.col(i);
          }
        }
        return true;
        */
    }
}
