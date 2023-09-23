crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/clamp.c]

pub fn pytorch_qnnp_create_clamp_nc_u8(
        channels:   usize,
        output_min: u8,
        output_max: u8,
        flags:      u32,
        clamp_out:  *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t clamp_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_clamp_nc_u8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create Clamp operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (output_min > output_max) {
        pytorch_qnnp_log_error(
            "failed to create Clamp operator with [%" PRIu8 ", %" PRIu8
            "] output range: range min must be below range max",
            output_min,
            output_max);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      clamp_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (clamp_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      clamp_op->channels = channels;
      clamp_op->u8_clamping_params =
          pytorch_qnnp_compute_u8_clamping_params(output_min, output_max);

      clamp_op->ukernel_type = pytorch_qnnp_ukernel_type_clamp;
      clamp_op->format = pytorch_qnnp_format_quint8;

      *clamp_out = clamp_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(clamp_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_clamp_nc_u8(
        clamp:         PyTorchQnnpOperator,
        batch_size:    usize,
        input:         *const u8,
        input_stride:  usize,
        output:        *mut u8,
        output_stride: usize) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_clamp_nc_u8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        clamp->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      clamp->batch_size = batch_size;
      clamp->input = input;
      clamp->input_pixel_stride = input_stride;
      clamp->output = output;
      clamp->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
