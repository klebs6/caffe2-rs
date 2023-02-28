crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/add.c]

pub fn pytorch_qnnp_create_add_nc_q8(
        channels:       Size,
        a_zero_point:   u8,
        a_scale:        f32,
        b_zero_point:   u8,
        b_scale:        f32,
        sum_zero_point: u8,
        sum_scale:      f32,
        sum_min:        u8,
        sum_max:        u8,
        flags:          u32,
        add_out:        *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t add_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_add_nc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (a_scale <= 0.0f || !isnormal(a_scale)) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %.7g A scale: scale must be finite and positive",
            a_scale);
        goto error;
      }

      if (b_scale <= 0.0f || !isnormal(b_scale)) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %.7g B scale: scale must be finite and positive",
            b_scale);
        goto error;
      }

      if (sum_scale <= 0.0f || !isnormal(sum_scale)) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %.7g output scale: scale must be finite and positive",
            sum_scale);
        goto error;
      }

      if (sum_min >= sum_max) {
        pytorch_qnnp_log_error(
            "failed to create add operator with [%" PRIu8 ", %" PRIu8
            "] output range: range min must be below range max",
            sum_min,
            sum_max);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      const float a_output_scale = a_scale / sum_scale;
      if (a_output_scale < 0x1.0p-14f || a_output_scale >= 0x1.0p+8f) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
            a_output_scale);
        goto error;
      }

      const float b_output_scale = b_scale / sum_scale;
      if (b_output_scale < 0x1.0p-14f || b_output_scale >= 0x1.0p+8f) {
        pytorch_qnnp_log_error(
            "failed to create add operator with %.7g A-to-output scale ratio: scale ratio must be in [2**-14, 2**8) range",
            b_output_scale);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      add_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (add_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      add_op->channels = channels;
      add_op->add_quantization_params =
          pytorch_qnnp_compute_add_quantization_params(
              a_zero_point,
              b_zero_point,
              sum_zero_point,
              a_scale / sum_scale,
              b_scale / sum_scale,
              sum_min,
              sum_max);

      add_op->ukernel_type = pytorch_qnnp_ukernel_type_add;
      add_op->format = pytorch_qnnp_format_quint8;

      *add_out = add_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(add_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_add_nc_q8(
    add_op:     PyTorchQnnpOperator,
    batch_size: Size,
    a:          *const u8,
    a_stride:   Size,
    b:          *const u8,
    b_stride:   Size,
    sum:        *mut u8,
    sum_stride: Size) -> PyTorchQnnpStatus {

    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_add_nc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        add_op->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      add_op->batch_size = batch_size;
      add_op->input = a;
      add_op->input_pixel_stride = a_stride;
      add_op->input2 = b;
      add_op->input2_pixel_stride = b_stride;
      add_op->output = sum;
      add_op->output_pixel_stride = sum_stride;

      return pytorch_qnnp_status_success;
        */
}
