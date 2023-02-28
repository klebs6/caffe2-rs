crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cudnn/LossCTC.cpp]

/// See Note [ATen preprocessor philosophy]
///
#[cfg(any(not(AT_CUDNN_ENABLED),CUDNN_VERSION_LT_7600))]
pub fn use_cudnn_ctc_loss(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64) -> bool {
    
    todo!();
        /*
            return false;
        */
}

#[cfg(any(not(AT_CUDNN_ENABLED),CUDNN_VERSION_LT_7600))]
pub fn cudnn_ctc_loss(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64,
        deterministic:  bool,
        zero_infinity:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            AT_ERROR("cudnn_ctc_loss: ATen not compiled with cuDNN >= 7 support");
        */
}

#[cfg(not(any(not(AT_CUDNN_ENABLED),CUDNN_VERSION_LT_7600)))]
pub fn use_cudnn_ctc_loss(
        log_probs:      &Tensor,
        targets:        &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64) -> bool {
    
    todo!();
        /*
            auto& ctx = globalContext();

      bool use_cudnn = ctx.userEnabledCuDNN() && (BLANK == 0) &&
          (targets.dim() == 1) && (log_probs.scalar_type() == kFloat) &&
          (targets.scalar_type() == kInt) &&
          (log_probs.device().type() == kCUDA);

      if (use_cudnn) {
        // we don't know that input_lengths and target_lengths have the same size
        // (they should, but we didn't check yet)
        i64 max_input_length = log_probs.size(0);
        for (usize b = 0; b < input_lengths.size(); b++) {
          use_cudnn &= (input_lengths[b] == max_input_length);
        }
        for (usize b = 0; b < target_lengths.size(); b++) {
          // target length < 256 is documented, but we see illegal memory accesses
          // when target lengths > input lengths for CuDNN
          use_cudnn &=
              (target_lengths[b] < 256) & (target_lengths[b] <= input_lengths[b]);
        }
      }
      return use_cudnn;
        */
}

#[cfg(not(any(not(AT_CUDNN_ENABLED),CUDNN_VERSION_LT_7600)))]
pub fn cudnn_ctc_loss(
        log_probs_t:    &Tensor,
        targets_t:      &Tensor,
        input_lengths:  &[i32],
        target_lengths: &[i32],
        BLANK:          i64,
        deterministic:  bool,
        zero_infinity:  bool) -> (Tensor,Tensor) {
    
    todo!();
        /*
            (void)zero_infinity; // only used for backward
      CheckedFrom c = "cudnn_ctc_loss";
      TensorArg log_probs { log_probs_t, "log_probs", 1 };
      TensorArg targets { targets_t, "targets", 2 };
      checkDim(c, log_probs, 3);
      checkScalarType(c, log_probs, kFloat);
      checkDim(c, targets, 1);
      checkScalarType(c, targets, kInt);
      checkContiguous(c, targets); // ?
      checkBackend(c, {*log_probs}, Backend::CUDA);
      checkBackend(c, {*targets}, Backend::CPU);
      i64 batch_size = log_probs->size(1);
      TORCH_CHECK(input_lengths_.size() == batch_size, "input_lengths needs to have size to match batch_size");
      TORCH_CHECK(target_lengths_.size() == batch_size, "target_lengths needs to have size to match batch_size");

      vector<int> input_lengths(input_lengths_.begin(), input_lengths_.end());
      vector<int> target_lengths(target_lengths_.begin(), target_lengths_.end());

      TORCH_CHECK(BLANK == 0, "blank must be label 0 for cudnn_ctc_loss");
      // checked in dispatch:
      // assert other conditions for cudnnCTCLoss: all label lengths <= 256
      // all input lengths = logprob.size(0)

      auto handle = getCudnnHandle();

      cudnnCTCLossAlgo_t algo = (deterministic ? CUDNN_CTC_LOSS_ALGO_DETERMINISTIC : CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC);

      CTCLossDescriptor ctc_loss_desc;

      // so the CuDNN gradient semantics have changed between 7.1 and 7.6,
      // this is CuDNN 7.6 only, see PyTorch 1.2 for older CuDNN.
      ctc_loss_desc.setEx(
          CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN);
      TensorDescriptor log_probs_desc{log_probs_t};
      Tensor grad = empty_like(log_probs_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      TensorDescriptor grad_desc{grad};

      usize workspace_size;
      AT_CUDNN_CHECK(cudnnGetCTCLossWorkspaceSize(
          handle,
          log_probs_desc.desc(),
          grad_desc.desc(),
          targets->data_ptr<int>(),
          target_lengths.data(),
          input_lengths.data(),
          algo,
          ctc_loss_desc.desc(),
          &workspace_size));

      Tensor workspace = empty(workspace_size, log_probs->options().dtype(kByte));
      Tensor costs = empty({log_probs->size(1)}, log_probs->options());

      AT_CUDNN_CHECK(cudnnCTCLoss(
          handle,
          log_probs_desc.desc(),
          log_probs_t.data_ptr(),
          targets->data_ptr<int>(),
          target_lengths.data(),
          input_lengths.data(),
          costs.data_ptr(),
          grad_desc.desc(),
          grad.data_ptr(),
          algo,
          ctc_loss_desc.desc(),
          workspace.data_ptr(),
          workspace_size));
      return make_tuple(costs, grad);
        */
}
