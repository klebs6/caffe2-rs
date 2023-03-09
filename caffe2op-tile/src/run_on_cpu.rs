crate::ix!();

impl TileOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
          at::Half,
          std::uint8_t,
          std::int32_t,
          std::int64_t,
          float,
          double,
          std::string>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type_string(&mut self) -> bool {
        
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
        if (Input(1).IsType<std::int64_t>()) {
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

      const TypeMeta meta = X.dtype();
      const int item_size = X.itemsize();
      const char* X_ptr = reinterpret_cast<const char*>(X.raw_data());
      char* Y_ptr = reinterpret_cast<char*>(Y->raw_mutable_data(meta));
      for (int i = 0; i < outer_size; ++i) {
        for (int t = 0; t < tiles_; ++t) {
          context_.CopyItemsSameDevice(meta, inner_size, X_ptr, Y_ptr);
          Y_ptr += inner_size * item_size;
        }
        X_ptr += inner_size * item_size;
      }
      return true;
        */
    }
}
