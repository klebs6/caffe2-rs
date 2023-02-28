// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/leaky-relu.c]

pub fn pytorch_qnnp_create_leaky_relu_nc_q8(
    channels:          Size,
    negative_slope:    f32,
    input_zero_point:  u8,
    input_scale:       f32,
    output_zero_point: u8,
    output_scale:      f32,
    output_min:        u8,
    output_max:        u8,
    flags:             u32,
    leaky_relu_out:    *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {

    todo!();
    /*
       pytorch_qnnp_operator_t leaky_relu_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (negative_slope <= 0.0f || !isnormal(negative_slope)) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %.7g negative slope: slope must be finite and positive",
            negative_slope);
        goto error;
      }

      if (negative_slope > 1.0f) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %.7g negative slope: slope must not exceed 1.0",
            negative_slope);
        goto error;
      }

      if (input_scale <= 0.0f || !isnormal(input_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %.7g input scale: scale must be finite and positive",
            input_scale);
        goto error;
      }

      if (output_scale <= 0.0f || !isnormal(output_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %.7g output scale: scale must be finite and positive",
            output_scale);
        goto error;
      }

      if (output_min >= output_max) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with [%" PRIu8 ", %" PRIu8
            "] output range: range min must be below range max",
            output_min,
            output_max);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      const float input_output_scale = input_scale / output_scale;
      if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
        pytorch_qnnp_log_error(
            "failed to create Leaky ReLU operator with %.7g input-to-output scale ratio: "
            "scale ratio must be in [2**-8, 2**8) range",
            input_output_scale);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      leaky_relu_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (leaky_relu_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      leaky_relu_op->lookup_table = malloc(256 * sizeof(u8));
      if (leaky_relu_op->lookup_table == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate 256 bytes for Leaky ReLU lookup table");
        goto error;
      }

      u8* lookup_table = leaky_relu_op->lookup_table;
      const float scaled_min_less_zero_point =
          (float)((i32)output_min - (i32)output_zero_point);
      const float scaled_max_less_zero_point =
          (float)((i32)output_max - (i32)output_zero_point);
      for (i32 i = 0; i < 256; i++) {
        const float x =
            input_output_scale * (float)(i - (i32)(u32)input_zero_point);
        float y = x < 0.0f ? x * negative_slope : x;
        if (y < scaled_min_less_zero_point) {
          y = scaled_min_less_zero_point;
        }
        if (y > scaled_max_less_zero_point) {
          y = scaled_max_less_zero_point;
        }
        lookup_table[(u32)i] = (u8)(lrintf(y) + (long)output_zero_point);
      }

      leaky_relu_op->channels = channels;

      leaky_relu_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
      leaky_relu_op->format = pytorch_qnnp_format_quint8;

      *leaky_relu_out = leaky_relu_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(leaky_relu_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_leaky_relu_nc_q8(
    leaky_relu:    PyTorchQnnpOperator,
    batch_size:    Size,
    input:         *const u8,
    input_stride:  Size,
    output:        *mut u8,
    output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_leaky_relu_nc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        leaky_relu->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      leaky_relu->batch_size = batch_size;
      leaky_relu->input = input;
      leaky_relu->input_pixel_stride = input_stride;
      leaky_relu->output = output;
      leaky_relu->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
