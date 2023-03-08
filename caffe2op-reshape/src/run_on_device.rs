crate::ix!();

impl<F,Context> ReshapeOp<F,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            new_shape_(this->template GetRepeatedArgument<int64_t>("shape"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 2) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        }
        CAFFE_ENFORCE(
            OperatorStorage::HasArgument("shape"), "Argument `shape` is missing.");
        return this->template DoRunWithType<int64_t>();
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            DoRunWithTypeImpl<T>(Input(0), Output(0));
        return true;
        */
    }
    
    #[inline] pub fn do_run_with_type_impl<T>(&mut self, input: &Tensor, output: *mut Tensor)  {
    
        todo!();
        /*
            vector<int64_t> actual_new_shape = new_shape_;
        if (InputSize() == 2) {
          CAFFE_ENFORCE(
              !OperatorStorage::HasArgument("shape"),
              "New shape is specified by the input blob, do not pass in "
              "the argument `shape`.");

          // Shape should be always stored only on CPU
          // Just in case if for some reason shape is on GPU
          if (this->InputIsTensorType(1, CPU)) {
            // originally, shape input must be in CPU context
            auto& shape = this->template Input<Tensor>(1, CPU);
            CAFFE_ENFORCE_EQ(
                shape.dim(),
                1,
                "When input_as_shape is true, the input must be a 1D tensor of "
                "data type int64_t");
            CAFFE_ENFORCE(shape.numel() > 0);
            auto* shape_data = shape.template data<T>();
            actual_new_shape.insert(
                actual_new_shape.end(), shape_data, shape_data + shape.dim32(0));
          } else {
            auto& shape = Input(1);
            CAFFE_ENFORCE_EQ(
                shape.dim(),
                1,
                "When input_as_shape is true, the input must be a 1D tensor of "
                "data type int64_t");
            CAFFE_ENFORCE(shape.numel() > 0);
            auto* shape_data = shape.template data<T>();
            // Fetch copy from
            std::unique_ptr<T[]> shape_data_copy =
                std::make_unique<T[]>(shape.dim32(0));
            context_.template CopyToCPU<T>(
                shape.dim32(0), shape_data, shape_data_copy.get());
            actual_new_shape.insert(
                actual_new_shape.end(),
                shape_data_copy.get(),
                shape_data_copy.get() + shape.dim32(0));
          }
        }

        // Checks if the new shape is valid and fills in the missing dimension
        // specified by -1.
        // NOTE: At most one dimension can be -1.
        auto total_size = input.numel();
        T size = 1;

        // NOTE: support for legacy caffe1 syntax
        // Copy over the dimensions for those that are specified zero.
        if (total_size != 0) {
          for (size_t i = 0; i < actual_new_shape.size() && i < input.dim(); ++i) {
            if (actual_new_shape[i] == 0) {
              actual_new_shape[i] = input.size(i);
            }
          }
        }

        int unknown_idx = -1;
        for (int i = 0; i < actual_new_shape.size(); ++i) {
          const auto dim = actual_new_shape[i];
          if (dim == -1) {
            CAFFE_ENFORCE(
                unknown_idx == -1,
                "Argument `shape` has more than one missing dimension.");
            unknown_idx = i;
          } else {
            size *= dim;
          }
        }
        if (size == 0 && total_size != 0) {
          CAFFE_THROW(
              "Can not reshape a non-zero size (",
              total_size,
              ") tensor to zero size.");
        }
        if (total_size != 0) {
          // if tensor is not empty, infer the size of the unknown index
          if (unknown_idx != -1) {
            CAFFE_ENFORCE_NE(
                size,
                0,
                "New shape at dim ",
                unknown_idx,
                " can not be inferred since new size is zero.");
            CAFFE_ENFORCE(
                total_size % size == 0,
                "Argument `shape` does not agree with the input data.",
                " (",
                total_size,
                " vs ",
                size,
                ")");
            actual_new_shape[unknown_idx] = total_size / size;
          } else {
            CAFFE_ENFORCE_EQ(
                total_size,
                size,
                "Argument `shape` does not agree with the input data.",
                " (",
                total_size,
                " != ",
                size,
                ")");
          }
        } else if (unknown_idx != -1) {
          // if size is empty, then set unknown index to be 0 (empty tensor)
          actual_new_shape[unknown_idx] = 0;
        }

        // Write the original shape to the second output.
        auto* old_shape = this->template Output<Tensor>(1, CPU);
        old_shape->Resize(input.sizes().size());
        T* old_shape_data = old_shape->template mutable_data<T>();
        std::vector<T> old_shape_vector(input.sizes().begin(), input.sizes().end());
        for (int i = 0; i < old_shape_vector.size(); ++i) {
          old_shape_data[i] = old_shape_vector[i];
        }

        output->Resize(actual_new_shape);
        if (output != &input) {
          // If we are not doing in-place computation, a copy is needed.
          context_.CopyItemsSameDevice(
              input.dtype(),
              input.numel(),
              input.raw_data(),
              output->raw_mutable_data(input.dtype()));
        }
        */
    }
}
