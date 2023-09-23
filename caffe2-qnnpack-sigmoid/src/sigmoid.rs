// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/sigmoid.c]

pub fn pytorch_qnnp_create_sigmoid_nc_q8(
    channels:          Size,
    input_zero_point:  u8,
    input_scale:       f32,
    output_zero_point: u8,
    output_scale:      f32,
    output_min:        u8,
    output_max:        u8,
    flags:             u32,
    sigmoid_out:       *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t sigmoid_op = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with %zu channels: number of channels must be non-zero",
            channels);
        goto error;
      }

      if (input_scale <= 0.0f || !isnormal(input_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with %.7g input scale: scale must be finite and positive",
            input_scale);
        goto error;
      }

      if (output_scale <= 0.0f || !isnormal(output_scale)) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with %.7g output scale: scale must be finite and positive",
            output_scale);
        goto error;
      }

      if (output_min >= output_max) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with [%" PRIu8 ", %" PRIu8
            "] output range: range min must be below range max",
            output_min,
            output_max);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      if (output_scale != 0x1.0p-8f) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with %.7g output scale: only output scale of 1/256 is supported",
            output_scale);
        goto error;
      }

      if (output_zero_point != 0) {
        pytorch_qnnp_log_error(
            "failed to create Sigmoid operator with %" PRIu8
            " output zero point: only output zero point of 0 is supported",
            output_zero_point);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      sigmoid_op = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (sigmoid_op == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      sigmoid_op->lookup_table = malloc(256 * sizeof(u8));
      if (sigmoid_op->lookup_table == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate 256 bytes for Sigmoid lookup table");
        goto error;
      }

      u8* lookup_table = sigmoid_op->lookup_table;
      const float scaled_min = (float)(i32)output_min;
      const float scaled_max = (float)(i32)output_max;
      for (i32 i = 0; i < 256; i++) {
        const float x =
            input_scale * (float)(i - (i32)(u32)input_zero_point);
        /* Scale sigmoid(x) by 1 / output scale = 256.0 */
        float scaled_sigmoid_x = 256.0f / (1.0f + expf(-x));
        if (scaled_sigmoid_x < scaled_min) {
          scaled_sigmoid_x = scaled_min;
        }
        if (scaled_sigmoid_x > scaled_max) {
          scaled_sigmoid_x = scaled_max;
        }
        lookup_table[(u32)i] = (u8)lrintf(scaled_sigmoid_x);
      }

      sigmoid_op->channels = channels;

      sigmoid_op->ukernel_type = pytorch_qnnp_ukernel_type_lut;
      sigmoid_op->format = pytorch_qnnp_format_quint8;

      *sigmoid_out = sigmoid_op;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(sigmoid_op);
      return status;
        */
}

pub fn pytorch_qnnp_setup_sigmoid_nc_q8(
    sigmoid:       PyTorchQnnpOperator,
    batch_size:    Size,
    input:         *const u8,
    input_stride:  Size,
    output:        *mut u8,
    output_stride: Size) -> PyTorchQnnpStatus {

    todo!();
    /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_sigmoid_nc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        sigmoid->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      sigmoid->batch_size = batch_size;
      sigmoid->input = input;
      sigmoid->input_pixel_stride = input_stride;
      sigmoid->output = output;
      sigmoid->output_pixel_stride = output_stride;

      return pytorch_qnnp_status_success;
        */
}
