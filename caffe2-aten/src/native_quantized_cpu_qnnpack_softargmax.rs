// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/softargmax.c]

pub fn pytorch_qnnp_create_softargmax_nc_q8(
    channels:          Size,
    input_scale:       f32,
    output_zero_point: u8,
    output_scale:      f32,
    flags:             u32,
    softargmax_out:    *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t softargmax_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create Soft ArgMax operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (input_scale <= 0.0f || !isnormal(input_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Soft ArgMax operator with %.7g input scale: scale must be finite and positive",
            input_scale);
        goto error;
      }

      if (output_scale <= 0.0f || !isnormal(output_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Soft ArgMax operator with %.7g output scale: scale must be finite and positive",
            output_scale);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      if (output_scale != 0x1.0p-8f) {
        pytorch_qnnp_log_error(
            "failed to create Soft ArgMax operator with %.7g output scale: only output scale of 1/256 is supported",
            output_scale);
        goto error;
      }

      if (output_zero_point != 0) {
        pytorch_qnnp_log_error(
            "failed to create Soft ArgMax operator with %" PRIu8
            " output zero point: only output zero point of 0 is supported",
            output_zero_point);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      softargmax_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (softargmax_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      softargmax_op->lookup_table = malloc(256 * sizeof(u32));
      if (softargmax_op->lookup_table == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate 256 bytes for Soft ArgMax lookup table");
        goto error;
      }

      u32* lookup_table = softargmax_op->lookup_table;
      const double qscale =
          fmin(((double)UINT32_MAX) / (double)channels, 8388607.0);
      for (i32 i = 0; i < 256; i++) {
        const double scaled_exp_xi =
            qscale * exp((double)(i - 255) * (double)input_scale);
        lookup_table[(u32)i] = (u32)lrint(scaled_exp_xi);
      }

      softargmax_op->channels = channels;

      softargmax_op->ukernel_type = pytorch_qnnp_ukernel_type_softargmax;
      softargmax_op->format = pytorch_qnnp_format_quint8;

      *softargmax_out = softargmax_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(softargmax_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_softargmax_nc_q8(
    softargmax:    PyTorchQnnpOperator,
    batch_size:    Size,
    input:         *const u8,
    input_stride:  Size,
    output:        *mut u8,
    output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_softargmax_nc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        softargmax->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      softargmax->batch_size = batch_size;
      softargmax->input = input;
      softargmax->input_pixel_stride = input_stride;
      softargmax->output = output;
      softargmax->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
