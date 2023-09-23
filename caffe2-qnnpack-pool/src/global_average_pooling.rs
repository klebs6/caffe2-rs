crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/global-average-pooling.c]

pub fn pytorch_qnnp_create_global_average_pooling_nwc_q8(
    channels:                   Size,
    input_zero_point:           u8,
    input_scale:                f32,
    output_zero_point:          u8,
    output_scale:               f32,
    output_min:                 u8,
    output_max:                 u8,
    flags:                      u32,
    global_average_pooling_out: *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t global_average_pooling_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create global average pooling operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (input_scale <= 0.0f || !isnormal(input_scale)) {
        pytorch_qnnp_log_error(
            "failed to create global average pooling operator with %.7g input scale: scale must be finite and positive",
            input_scale);
        goto error;
      }

      if (output_scale <= 0.0f || !isnormal(output_scale)) {
        pytorch_qnnp_log_error(
            "failed to create global average pooling operator with %.7g output scale: scale must be finite and positive",
            output_scale);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      const float input_output_scale = input_scale / output_scale;
      if (input_output_scale < 0x1.0p-8f || input_output_scale >= 0x1.0p+8f) {
        pytorch_qnnp_log_error(
            "failed to create global average pooling operator with %.7g input-to-output scale ratio: "
            "scale ratio must be in [2**-8, 2**8) range",
            input_output_scale);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      global_average_pooling_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (global_average_pooling_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      void* zero_buffer = calloc(channels, sizeof(u8));
      if (zero_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for zero padding",
            channels * sizeof(u8));
        goto error;
      }
      global_average_pooling_op->zero_buffer = zero_buffer;
      global_average_pooling_op->zero_pointer = zero_buffer;

      global_average_pooling_op->channels = channels;
      global_average_pooling_op->input_zero_point = input_zero_point;
      global_average_pooling_op->output_zero_point = output_zero_point;
      global_average_pooling_op->input_scale = input_scale;
      global_average_pooling_op->output_scale = output_scale;
      global_average_pooling_op->output_min = output_min;
      global_average_pooling_op->output_max = output_max;

      global_average_pooling_op->ukernel_type =
          pytorch_qnnp_ukernel_type_global_average_pooling;
      global_average_pooling_op->format = pytorch_qnnp_format_quint8;

      *global_average_pooling_out = global_average_pooling_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(global_average_pooling_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_global_average_pooling_nwc_q8(
    global_average_pooling_op: PyTorchQnnpOperator,
    batch_size:                Size,
    width:                     Size,
    input:                     *const u8,
    input_stride:              Size,
    output:                    *mut u8,
    output_stride:             Size) -> PyTorchQnnpStatus {

    todo!();
    /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_global_average_pooling_nwc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        global_average_pooling_op->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      if (width == 0) {
        pytorch_qnnp_log_error(
            "failed to setup global average pooling operator with width %zu: width must be non-zero",
            width);
        return pytorch_qnnp_status_invalid_parameter;
      }

      global_average_pooling_op->batch_size = batch_size;
      global_average_pooling_op->input_width = width;
      global_average_pooling_op->input = input;
      global_average_pooling_op->input_pixel_stride = input_stride;
      global_average_pooling_op->output = output;
      global_average_pooling_op->output_pixel_stride = output_stride;

      global_average_pooling_op->avgpool_quantization_params =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -(i32)width *
                  (i32)(u32)global_average_pooling_op->input_zero_point,
              global_average_pooling_op->input_scale /
                  (global_average_pooling_op->output_scale * (float)width),
              global_average_pooling_op->output_zero_point,
              global_average_pooling_op->output_min,
              global_average_pooling_op->output_max);

      return pytorch_qnnp_status_success;
        */
}
