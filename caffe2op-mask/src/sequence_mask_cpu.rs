crate::ix!();

impl SequenceMaskOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<>(&mut self) -> bool {
        todo!();
        /*
            const Tensor* input = &Input(0);
          const Tensor* sequence_lengths = nullptr;
          const Tensor* window_centers = nullptr;

          if (mode_ == "sequence") {
            sequence_lengths = &Input(1);
          } else if (mode_ == "window") {
            window_centers = &Input(1);
          }

          auto* output = Output(0, input->sizes(), at::dtype<T>());

          const auto canonical_axis = input->canonical_axis_index(axis_);

          // canonical_batch is non-negative if batching, -1 otherwise
          int canonical_batch = -1;
          if ((HasArgument("batch"))) {
            canonical_batch = input->canonical_axis_index(batch_);
          }

          // make sure batch < axis
          if (canonical_batch >= 0) {
            CAFFE_ENFORCE_LT(canonical_batch, canonical_axis);
          }

          // if no batch, then left is product of dims up to axis
          // otherwise, left is product of dims between batch and axis
          const int left =
              (canonical_batch >= 0
                   ? input->size_between_dim(canonical_batch, canonical_axis)
                   : input->size_to_dim(canonical_axis));
          const int right = input->size_from_dim(canonical_axis);

          // product of dims from 1 to batch
          const int batch_dim =
              (canonical_batch >= 0
                   ? input->size_to_dim(canonical_batch) * input->size(canonical_batch)
                   : -1);

          T fill_val = convert::To<float, T>(grad_ ? 0.0f : fill_val_);
          if (mode_ == "sequence") {
            CAFFE_ENFORCE(
                sequence_lengths, "Sequence length not provided for mode 'sequence'!");
            if (HasArgument("repeat_from_axis")) {
              const int canonical_repeat_from =
                  input->canonical_axis_index(repeat_from_);
              const int repeated_dims = input->size_from_dim(canonical_repeat_from);
              const int masked_dims = right / repeated_dims;
              RepeatedMaskWithFunctor(
                  left,
                  masked_dims,
                  repeated_dims,
                  input->data<T>(),
                  SequenceFunctor(
                      sequence_lengths->data<int>(), sequence_lengths->numel()),
                  fill_val,
                  output->template mutable_data<T>());
            } else {
              MaskWithFunctor(
                  left,
                  right,
                  batch_dim,
                  input->data<T>(),
                  SequenceFunctor(
                      sequence_lengths->data<int>(), sequence_lengths->numel()),
                  fill_val,
                  output->template mutable_data<T>());
            }
          } else if (mode_ == "window") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                WindowFunctor(window_centers->data<int>(), radius_),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "upper") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                UpperFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "lower") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                LowerFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "upperdiag") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                UpperDiagFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else if (mode_ == "lowerdiag") {
            MaskWithFunctor(
                left,
                right,
                batch_dim,
                input->data<T>(),
                LowerDiagFunctor(),
                fill_val,
                output->template mutable_data<T>());
          } else {
            CAFFE_ENFORCE(false, "Unsupported mode for SequenceMaskOp!");
            return false;
          }

          return true;
        */
    }
}
