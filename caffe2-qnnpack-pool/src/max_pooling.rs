// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/max-pooling.c]

#[inline] pub fn compute_output_dimension(
    padded_input_dimension: usize,
    kernel_dimension:       usize,
    dilation_dimension:     usize,
    stride_dimension:       usize) -> usize {
    
    todo!();
        /*
            const usize effective_kernel_dimension =
          (kernel_dimension - 1) * dilation_dimension + 1;
      return (padded_input_dimension - effective_kernel_dimension) /
          stride_dimension +
          1;
        */
}

pub fn pytorch_qnnp_create_max_pooling2d_nhwc_u8(
    input_padding_top:    u32,
    input_padding_right:  u32,
    input_padding_bottom: u32,
    input_padding_left:   u32,
    pooling_height:       u32,
    pooling_width:        u32,
    stride_height:        u32,
    stride_width:         u32,
    dilation_height:      u32,
    dilation_width:       u32,
    channels:             usize,
    output_min:           u8,
    output_max:           u8,
    flags:                u32,
    max_pooling_out:      *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {

    todo!();
        /*
            pytorch_qnnp_operator_t max_pooling = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_max_pooling2d_nhwc_u8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      const u32 pooling_size = pooling_height * pooling_width;
      if (pooling_size == 0) {
        pytorch_qnnp_log_error(
            "failed to create max pooling with %" PRIu32 "x%" PRIu32
            " pooling size: "
            "pooling size dimensions must be non-zero",
            pooling_width,
            pooling_height);
        goto error;
      }

      if (pooling_size == 1) {
        pytorch_qnnp_log_error(
            "failed to create max pooling with 1 pooling element: "
            "1x1 pooling is meaningless");
        goto error;
      }

      if (stride_height == 0 || stride_width == 0) {
        pytorch_qnnp_log_error(
            "failed to create max pooling with %" PRIu32 "x%" PRIu32
            " stride: "
            "stride dimensions must be non-zero",
            stride_width,
            stride_height);
        goto error;
      }

      if (dilation_height == 0 || dilation_width == 0) {
        pytorch_qnnp_log_error(
            "failed to create max pooling with %" PRIu32 "x%" PRIu32
            " dilation: "
            "dilation dimensions must be non-zero",
            dilation_width,
            dilation_height);
        goto error;
      }

      if (channels == 0) {
        pytorch_qnnp_log_error(
            "failed to create max pooling with %zu channels: "
            "number of channels must be non-zero",
            channels);
        goto error;
      }

      status = pytorch_qnnp_status_out_of_memory;

      max_pooling = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (max_pooling == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      max_pooling->input_padding_top = input_padding_top;
      max_pooling->input_padding_right = input_padding_right;
      max_pooling->input_padding_bottom = input_padding_bottom;
      max_pooling->input_padding_left = input_padding_left;

      max_pooling->kernel_height = pooling_height;
      max_pooling->kernel_width = pooling_width;
      max_pooling->stride_height = stride_height;
      max_pooling->stride_width = stride_width;
      max_pooling->dilation_height = dilation_height;
      max_pooling->dilation_width = dilation_width;
      max_pooling->channels = channels;

      max_pooling->u8_clamping_params =
          pytorch_qnnp_compute_u8_clamping_params(output_min, output_max);

      max_pooling->ukernel_type = pytorch_qnnp_ukernel_type_max_pooling;
      max_pooling->format = pytorch_qnnp_format_quint8;

      *max_pooling_out = max_pooling;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(max_pooling);
      return status;
        */
}

pub fn pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
    max_pooling:         PyTorchQnnpOperator,
    batch_size:          usize,
    input_height:        usize,
    input_width:         usize,
    input:               *const u8,
    input_pixel_stride:  usize,
    output:              *mut u8,
    output_pixel_stride: usize,
    threadpool:          threadpool::ThreadPool) -> PyTorchQnnpStatus {

    todo!();
        /*
            if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_setup_max_pooling2d_nhwc_u8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        max_pooling->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      if (input_width == 0 || input_height == 0) {
        pytorch_qnnp_log_error(
            "failed to setup max pooling with %zux%zu input: input dimensions must be non-zero",
            input_width,
            input_height);
        return pytorch_qnnp_status_invalid_parameter;
      }

      max_pooling->batch_size = batch_size;
      max_pooling->input_height = input_height;
      max_pooling->input_width = input_width;
      max_pooling->input = input;
      max_pooling->input_pixel_stride = input_pixel_stride;

      max_pooling->output_height = compute_output_dimension(
          max_pooling->input_padding_top + input_height +
              max_pooling->input_padding_bottom,
          max_pooling->kernel_height,
          max_pooling->dilation_height,
          max_pooling->stride_height);
      max_pooling->output_width = compute_output_dimension(
          max_pooling->input_padding_left + input_width +
              max_pooling->input_padding_right,
          max_pooling->kernel_width,
          max_pooling->dilation_width,
          max_pooling->stride_width);
      max_pooling->output = output;
      max_pooling->output_pixel_stride = output_pixel_stride;

      usize valid_batch_size = 0;
      if (input == max_pooling->last_input &&
          input_height == max_pooling->last_input_height &&
          input_width == max_pooling->last_input_width) {
        valid_batch_size = max_pooling->valid_batch_size;
        if (batch_size <= valid_batch_size) {
          return pytorch_qnnp_status_success;
        }
      }

      const usize pooling_height = max_pooling->kernel_height;
      const usize pooling_width = max_pooling->kernel_width;
      const usize pooling_size = pooling_height * pooling_width;
      const usize output_height = max_pooling->output_height;
      const usize output_width = max_pooling->output_width;
      /* Micro-kernel may read up to (mr - 1) elements after the end of indirection
       * buffer */
      const u32 mr = pytorch_qnnp_params.u8maxpool.mr;

      const usize step_width = max_pooling->dilation_width > 1
          ? pooling_width
          : min(max_pooling->stride_width, pooling_width);
      const usize step_height =
          pooling_size + (output_width * step_width - 1) * pooling_height;
      const usize indirection_buffer_size =
          sizeof(void*) * ((mr - 1) + batch_size * output_height * step_height);

      const void** indirection_buffer = (const void**)realloc(
          max_pooling->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      max_pooling->indirection_buffer = indirection_buffer;

      pytorch_qnnp_indirection_init_maxpool2d(
          max_pooling, valid_batch_size, step_height, step_width);

      max_pooling->last_input = input;
      max_pooling->last_input_height = input_height;
      max_pooling->last_input_width = input_width;
      max_pooling->valid_batch_size = max(valid_batch_size, batch_size);

      return pytorch_qnnp_status_success;
        */
}
