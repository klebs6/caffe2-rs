crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qchannel_shuffle.cpp]

#[cfg(USE_PYTORCH_QNNPACK)]
pub fn quantized_channel_shuffle_impl(
        self_:  &Tensor,
        groups: i64) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          groups > 0,
          "Number of groups to divide channels in must be positive.",
          " Value of groups:", groups);
      TORCH_CHECK(
          self.dim() == 4,
          "channel_shuffle expects 4D input, but got input with sizes ",
          self.sizes());
      TORCH_CHECK(
          self.scalar_type() == kQUInt8,
          "Quantized channel shuffle works only on u8.",
          "But got:", self.scalar_type());
      const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
      Tensor qy = native::empty_affine_quantized(
          self_nhwc.sizes(),
          kQUInt8,
          nullopt /* layout */,
          kCPU,
          nullopt /* pin_memory */,
          self_nhwc.q_scale(),
          self_nhwc.q_zero_point(),
          MemoryFormat::ChannelsLast);

      // Degenerate case of just copying.
      if (groups == 1) {
        qy.copy_(self_nhwc);
        return qy.contiguous(self.suggest_memory_format());
      }

      i64 channels = self.size(1);
      TORCH_CHECK(channels > 0,
                 "Number of channels must be positive, got:", channels);
      TORCH_CHECK((channels % groups) == 0,
                 "Number of channels must be divisible gy groups. Got ",
                 channels, " channels and ", groups, " groups.");

      initQNNPACK();

      pytorch_qnnp_operator_t qnnpack_operator{nullptr};

      const pytorch_qnnp_status createStatus = pytorch_qnnp_create_channel_shuffle_nc_x8(
          groups /* groups */,
          channels / groups /* group channels */,
          0 /* flags */,
          &qnnpack_operator);
      TORCH_INTERNAL_ASSERT(
          createStatus == pytorch_qnnp_status_success,
          "failed to create QNNPACK ChannelShuffle operator");

      unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
          qnnpack_uniq_ptr(qnnpack_operator);

      const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_channel_shuffle_nc_x8(
          qnnpack_uniq_ptr.get(),
          self_nhwc.numel() / channels /* batch size */,
          (u8*)self_nhwc.data_ptr<quint8>() /* self data */,
          channels /* self stride */,
          (u8*)qy.data_ptr<quint8>() /* qy data */,
          channels /* qy stride */);
      TORCH_INTERNAL_ASSERT(
          setupStatus == pytorch_qnnp_status_success,
          "failed to setup QNNPACK ChannelShuffle operator");

      pthreadpool_t threadpool = pthreadpool_();
      const pytorch_qnnp_status runStatus =
          pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
      TORCH_INTERNAL_ASSERT(
          runStatus == pytorch_qnnp_status_success,
          "failed to run QNNPACK ChannelShuffle operator");

      return qy.contiguous(self.suggest_memory_format());
        */
}

/// native functions for the native_functions.yaml
///
pub fn channel_shuffle_quantized_cpu(
        self_:  &Tensor,
        groups: i64) -> Tensor {
    
    todo!();
        /*
            #ifdef USE_PYTORCH_QNNPACK
      return quantized_channel_shuffle_impl(self, groups);
    #endif
      // If QNNPACK is not available then fall back to the
      // non quantized path.
      return native::channel_shuffle(self, groups);
        */
}

/// Keep the registry in the anonymous namespace.
///
pub struct QChannelShuffle {
    base: OperatorKernel,
}

impl QChannelShuffle {
    
    pub fn invoke(&mut self, 
        qx:     Tensor,
        groups: i64) -> Tensor {
        
        todo!();
        /*
            return channel_shuffle_quantized_cpu(qx, groups);
        */
    }
}
