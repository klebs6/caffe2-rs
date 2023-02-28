crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected.c]

pub fn pytorch_qnnp_create_fully_connected_nc_q8(
    input_channels:        Size,
    output_channels:       Size,
    input_zero_point:      u8,
    kernel_zero_points:    *const u8,
    kernel:                *const u8,
    bias:                  *const i32,
    output_zero_point:     u8,
    output_min:            u8,
    output_max:            u8,
    flags:                 u32,
    requantization_scales: *const f32,
    fully_connected_out:   *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {

    todo!();
        /*
            pytorch_qnnp_operator_t fully_connected = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      for (int i = 0; i < output_channels; ++i) {
        if (requantization_scales[i] <= 0.0f ||
            !isnormal(requantization_scales[i])) {
          pytorch_qnnp_log_error(
              "failed to create fully connected operator with %.7g requantization scale: scale must be finite and positive",
              requantization_scales[i]);
          goto error;
        }
      }

      status = pytorch_qnnp_status_out_of_memory;

      fully_connected = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (fully_connected == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      const u32 nr = pytorch_qnnp_params.q8conv.nr;
      const u32 kr = pytorch_qnnp_params.q8conv.kr;

      const u32 n_stride = (output_channels + (nr - 1)) & -nr;
      const u32 k_stride = (input_channels + (kr - 1)) & -kr;

      fully_connected->packed_weights =
          malloc(n_stride * (k_stride * sizeof(u8) + sizeof(i32)));
      if (fully_connected->packed_weights == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            n_stride * (k_stride * sizeof(u8) + sizeof(i32)));
        goto error;
      }
      memset(
          fully_connected->packed_weights,
          kernel_zero_points[0],
          n_stride * (k_stride * sizeof(u8) + sizeof(i32)));

      pytorch_pack_q8gemm_w(
          output_channels,
          input_channels,
          nr,
          nr,
          kr,
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          input_zero_point,
          kernel_zero_points[0],
    #endif
          kernel,
          bias,
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          kernel_zero_points,
    #endif
          fully_connected->packed_weights);

      fully_connected->groups = 1;
      fully_connected->group_input_channels = input_channels;
      fully_connected->group_output_channels = output_channels;

      fully_connected->kernel_zero_point = kernel_zero_points[0];

      fully_connected->conv_quantization_params =
          pytorch_qnnp_compute_conv_quantization_params(
              input_zero_point,
              kernel_zero_points,
              requantization_scales,
              output_zero_point,
              output_min,
              output_max);

      fully_connected->ukernel_type = pytorch_qnnp_ukernel_type_gemm;
      fully_connected->format = pytorch_qnnp_format_quint8;

      *fully_connected_out = fully_connected;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(fully_connected);
      return status;
        */
}

pub fn pytorch_qnnp_setup_fully_connected_nc_q8(
    fully_connected: PyTorchQnnpOperator,
    batch_size:      Size,
    input:           *const u8,
    input_stride:    Size,
    output:          *mut u8,
    output_stride:   Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_fully_connected_nc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        fully_connected->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      fully_connected->batch_size = 1;
      fully_connected->input_height = batch_size;
      fully_connected->input_width = 1;
      fully_connected->input = input;
      fully_connected->input_pixel_stride = input_stride;

      fully_connected->output_height = batch_size;
      fully_connected->output_width = 1;
      fully_connected->output = output;
      fully_connected->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
