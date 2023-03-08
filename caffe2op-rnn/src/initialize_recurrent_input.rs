crate::ix!();

/**
  | Copy external input to the step net into
  | the first item of (T + 1) X batch_size
  | X input_size tensor
  |
  */
#[inline] pub fn initialize_recurrent_input<T, Context>(
    rc:         &RecurrentInput,
    seq_len:    i32,
    batch_size: i32,
    ws:         *mut Workspace,
    context:    *mut Context) 
{
    todo!();
    /*
        auto stateBlob = ws->GetBlob(rc.state);
      CAFFE_ENFORCE(stateBlob);
      auto* state = BlobGetMutableTensor(stateBlob, Context::GetDeviceType());

      auto inputBlob = ws->GetBlob(rc.input);
      CAFFE_ENFORCE(inputBlob);
      const auto& input = inputBlob->template Get<Tensor>();
      CAFFE_ENFORCE_GE(input.dim(), 1, rc.input);
      CAFFE_ENFORCE_LE(input.dim(), 3, rc.input);

      const auto stateSize = input.size(input.dim() - 1);
      // Sometimes we want to provide more than one initial step.
      // For example, if we do a convolution op in step net
      // and need a sufficient left padding around the input.
      // This could be used together with links where window != 1.
      auto initialStateLength = 1;
      if (input.dim() == 3) {
        initialStateLength = input.size(0);
      }
      // States at [0, ..., (T + initialStateLength - 1)] (inclusive)
      state->Resize(seqLen + initialStateLength, batchSize, stateSize);

      if (input.dim() >= 2) {
        CAFFE_ENFORCE_EQ(input.size(input.dim() - 2), batchSize, rc.input);
        context->template CopySameDevice<T>(
            batchSize * stateSize * initialStateLength,
            input.template data<T>(),
            state->template mutable_data<T>());
      } else {
        // Usually, the initial state is the same for all inputs in the batch.
        // So the op conveniently accepts 1-D input and copies it batchSize times.
        repeatCopy<T, Context>(
              batchSize,
              stateSize,
              input.template data<T>(),
              state->template mutable_data<T>(),
              context);
      }
    */
}
