crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-prepack.cc]

impl PackBMatrix {
    
    /// For runtime quantization packing.
    ///
    pub fn new(
        input_channels:        usize,
        output_channels:       usize,
        kernel_zero_points:    *const u8,
        requantization_scales: *const f32,
        kernel:                *const u8,
        bias:                  *const i32) -> Self {
    
        todo!();
        /*


            for (usize i = 0; i < output_channels; ++i) {
        if (requantization_scales[i] <= 0.0f ||
            !isnormal(requantization_scales[i])) {
          pytorch_qnnp_log_error(
              "failed to create fully connected operator with requant scale of "
              "%.7g for output channel %d."
              "Scale must be finite and positive",
              requantization_scales[i], (int)i);
          assert("QNNPACK Runtime Error.");
        }
      }

      const u32 nr = pytorch_qnnp_params.q8conv.nr;
      const u32 kr = pytorch_qnnp_params.q8conv.kr;

      const u32 n_stride = (output_channels + (nr - 1)) & -nr;
      const u32 k_stride = (input_channels + (kr - 1)) & -kr;

      input_channels_ = input_channels;
      output_channels_ = output_channels;
      packed_weights_ =
          malloc(n_stride * (k_stride * sizeof(u8) + sizeof(i32)));
      if (packed_weights_ == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            n_stride * (k_stride * sizeof(u8) + sizeof(i32)));
        assert("QNNPACK Runtime Error.");
      }

      pytorch_pack_q8gemm_wrq(
          output_channels,
          input_channels,
          nr,
          nr,
          kr,
          kernel,
          bias,
          kernel_zero_points,
          packed_weights_);
        */
    }
}
