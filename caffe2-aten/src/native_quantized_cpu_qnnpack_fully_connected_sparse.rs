crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/fully-connected-sparse.c]

pub fn pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
    input_channels:        Size,
    output_channels:       Size,
    input_zero_point:      u8,
    kernel_zero_points:    *const u8,
    kernel_col_indices:    *const u32,
    kernel_row_values:     *const u32,
    kernel_values:         *const u8,
    kernel_row_block_size: u32,
    kernel_col_block_size: u32,
    output_zero_point:     u8,
    output_min:            u8,
    output_max:            u8,
    flags:                 u32,
    requantization_scales: *const f32,
    use_prepack_kernel:    bool,
    fully_connected_out:   *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t fully_connected = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8 failed because QNNPACK is not properly initialized");
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

      if (kernel_row_block_size == 8 && kernel_col_block_size == 1) {
        // This is to gate 8x1 on SSE2 since we have not implemented SSE2
        // kernel that suppors 8x1 sparsity pattern.
        if (pytorch_qnnp_params.q8gemm_sparse_c8x1.packA == NULL) {
          status = pytorch_qnnp_status_invalid_parameter;
          goto error;
        }
      }
      fully_connected->sparse_matrix.col_indices = kernel_col_indices;
      fully_connected->sparse_matrix.row_values = kernel_row_values;
      fully_connected->sparse_matrix.values = kernel_values;
      fully_connected->sparse_matrix.row_block_size = kernel_row_block_size;
      fully_connected->sparse_matrix.col_block_size = kernel_col_block_size;

      fully_connected->groups = 1;
      fully_connected->group_input_channels = input_channels;
      fully_connected->group_output_channels = output_channels;

      fully_connected->kernel_zero_point = kernel_zero_points[0];

      fully_connected->dynamic_conv_quantization_params.input_zero_point =
        input_zero_point;
      fully_connected->dynamic_conv_quantization_params.kernel_zero_points =
        kernel_zero_points;
      fully_connected->dynamic_conv_quantization_params.multipliers =
        requantization_scales;

      // Always use prepacking based kernel
      fully_connected->ukernel_type = pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq;
      fully_connected->format = pytorch_qnnp_format_quint8;

      *fully_connected_out = fully_connected;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(fully_connected);
      return status;
        */
}

pub fn pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
    fully_connected: PyTorchQnnpOperator,
    batch_size:      Size,
    input:           *const u8,
    input_stride:    Size,
    bias:            *const f32,
    output:          *mut f32,
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

      fully_connected->bias = bias;

      fully_connected->output_height = batch_size;
      fully_connected->output_width = 1;
      fully_connected->output = output;
      fully_connected->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
