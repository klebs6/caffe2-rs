crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/deconvolution.c]

#[inline] pub fn compute_output_dimension(
        input_dimension:         usize,
        input_padding_dimension: usize,
        adjustment_dimension:    usize,
        kernel_dimension:        usize,
        dilation_dimension:      usize,
        stride_dimension:        usize) -> usize {
    
    todo!();
        /*
            const usize effective_kernel_dimension =
          (kernel_dimension - 1) * dilation_dimension + 1;
      return stride_dimension * (input_dimension - 1) + adjustment_dimension +
          effective_kernel_dimension - input_padding_dimension;
        */
}

pub fn pytorch_qnnp_create_deconvolution2d_nhwc_q8(
        input_padding_top:     u32,
        input_padding_right:   u32,
        input_padding_bottom:  u32,
        input_padding_left:    u32,
        adjustment_height:     u32,
        adjustment_width:      u32,
        kernel_height:         u32,
        kernel_width:          u32,
        stride_height:         u32,
        stride_width:          u32,
        dilation_height:       u32,
        dilation_width:        u32,
        groups:                u32,
        group_input_channels:  usize,
        group_output_channels: usize,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        kernel:                *const u8,
        bias:                  *const i32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        flags:                 u32,
        requantization_scales: *const f32,
        deconvolution_out:     *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
            pytorch_qnnp_operator_t deconvolution = NULL;
      enum pytorch_qnnp_status status = pytorch_qnnp_status_uninitialized;

      if (!pytorch_qnnp_params.initialized) {
        pytorch_qnnp_log_error(
            "pytorch_qnnp_create_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
        goto error;
      }

      status = pytorch_qnnp_status_invalid_parameter;

      if (kernel_width == 0 || kernel_height == 0) {
        pytorch_qnnp_log_error(
            "failed to create deconvolution with %" PRIu32 "x%" PRIu32
            " kernel: kernel dimensions must be non-zero",
            kernel_width,
            kernel_height);
        goto error;
      }

      if (stride_width == 0 || stride_height == 0) {
        pytorch_qnnp_log_error(
            "failed to create deconvolution with %" PRIu32 "x%" PRIu32
            " stride: "
            "stride dimensions must be non-zero",
            stride_width,
            stride_height);
        goto error;
      }

      if (dilation_width == 0 || dilation_height == 0) {
        pytorch_qnnp_log_error(
            "failed to create deconvolution with %" PRIu32 "x%" PRIu32
            " dilation: "
            "dilation dimensions must be non-zero",
            dilation_width,
            dilation_height);
        goto error;
      }

      status = pytorch_qnnp_status_unsupported_parameter;

      for (int i = 0; i < groups * group_output_channels; i++) {
        if (requantization_scales[i] <= 0.0f ||
            !isnormal(requantization_scales[i])) {
          pytorch_qnnp_log_error(
              "failed to create deconvolution operator with %.7g requantization scale for "
              "channel %d scale must be finite and positive",
              requantization_scales[i], i);
          goto error;
        }
      }

      status = pytorch_qnnp_status_out_of_memory;

      deconvolution = calloc(1, sizeof(struct pytorch_qnnp_operator));
      if (deconvolution == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for pytorch_qnnp_operator structure",
            sizeof(struct pytorch_qnnp_operator));
        goto error;
      }

      const u32 nr = pytorch_qnnp_params.q8conv.nr;
      const u32 kr = pytorch_qnnp_params.q8conv.kr;

      const u32 n_stride = (group_output_channels + (nr - 1)) & -nr;
      const u32 k_stride = (group_input_channels + (kr - 1)) & -kr;
      const u32 kernel_size = kernel_height * kernel_width;
      const usize packed_group_weights_size =
          (sizeof(u8) * kernel_size * k_stride + sizeof(i32)) * n_stride;
      deconvolution->packed_weights = malloc(packed_group_weights_size * groups);
      if (deconvolution->packed_weights == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for packed weights",
            packed_group_weights_size * groups);
        goto error;
      }
      memset(
          deconvolution->packed_weights,
          kernel_zero_points[0],
          packed_group_weights_size * groups);

      for (u32 group = 0; group < groups; group++) {
        pytorch_pack_q8deconv_w(
            group_output_channels,
            kernel_size,
            group_input_channels,
            nr,
            kr,
    #if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            input_zero_point,
            kernel_zero_points[0],
    #endif
            kernel +
                group * group_output_channels * kernel_size * group_input_channels,
            bias + group * group_output_channels,
    #if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
            kernel_zero_points + group * group_output_channels,
    #endif
            (void*)((uintptr_t)deconvolution->packed_weights + group * packed_group_weights_size));
      }

      usize zero_size = sizeof(u8) * k_stride;
      usize zero_offset = 0;
      if (group_input_channels < 8) {
        zero_size += 8;
        zero_offset = 8;
      }

      void* zero_buffer = malloc(zero_size);
      if (zero_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for zero padding", zero_size);
        goto error;
      }
      memset(zero_buffer, input_zero_point, zero_size);
      deconvolution->zero_buffer = zero_buffer;
      deconvolution->zero_pointer = (void*)((uintptr_t)zero_buffer + zero_offset);

      deconvolution->input_padding_top = input_padding_top;
      deconvolution->input_padding_right = input_padding_right;
      deconvolution->input_padding_bottom = input_padding_bottom;
      deconvolution->input_padding_left = input_padding_left;
      deconvolution->adjustment_height = adjustment_height;
      deconvolution->adjustment_width = adjustment_width;

      deconvolution->kernel_height = kernel_height;
      deconvolution->kernel_width = kernel_width;
      deconvolution->stride_height = stride_height;
      deconvolution->stride_width = stride_width;
      deconvolution->dilation_height = dilation_height;
      deconvolution->dilation_width = dilation_width;
      deconvolution->groups = groups;
      deconvolution->group_input_channels = group_input_channels;
      deconvolution->group_output_channels = group_output_channels;

      deconvolution->kernel_zero_point = kernel_zero_points[0];

      deconvolution->conv_quantization_params =
          pytorch_qnnp_compute_conv_quantization_params(
              input_zero_point,
              kernel_zero_points,
              requantization_scales,
              output_zero_point,
              output_min,
              output_max);

      deconvolution->ukernel_type = pytorch_qnnp_ukernel_type_conv;
      deconvolution->format = pytorch_qnnp_format_quint8;

      *deconvolution_out = deconvolution;
      return pytorch_qnnp_status_success;

    error:
      pytorch_qnnp_delete_operator(deconvolution);
      return status;
        */
}

pub fn pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
    deconvolution:       PyTorchQnnpOperator,
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
            "pytorch_qnnp_setup_deconvolution2d_nhwc_q8 failed because QNNPACK is not properly initialized");
        return pytorch_qnnp_status_uninitialized;
      }

      if (batch_size == 0) {
        deconvolution->batch_size = 0;
        return pytorch_qnnp_status_success;
      }

      if (input_width == 0 || input_height == 0) {
        pytorch_qnnp_log_error(
            "failed to setup deconvolution with %zux%zu input: input dimensions must be non-zero",
            input_width,
            input_height);
        return pytorch_qnnp_status_invalid_parameter;
      }

      deconvolution->batch_size = batch_size;
      deconvolution->input_height = input_height;
      deconvolution->input_width = input_width;
      deconvolution->input = input;
      deconvolution->input_pixel_stride = input_pixel_stride;
      deconvolution->output = output;
      deconvolution->output_pixel_stride = output_pixel_stride;

      const usize kernel_height = deconvolution->kernel_height;
      const usize kernel_width = deconvolution->kernel_width;
      const usize kernel_size = kernel_height * kernel_width;
      const usize stride_height = deconvolution->stride_height;
      const usize stride_width = deconvolution->stride_width;
      const usize output_height = deconvolution->output_height =
          compute_output_dimension(
              input_height,
              deconvolution->input_padding_top +
                  deconvolution->input_padding_bottom,
              deconvolution->adjustment_height,
              kernel_height,
              deconvolution->dilation_height,
              stride_height);
      const usize output_width = deconvolution->output_width =
          compute_output_dimension(
              input_width,
              deconvolution->input_padding_left +
                  deconvolution->input_padding_right,
              deconvolution->adjustment_width,
              kernel_width,
              deconvolution->dilation_width,
              stride_width);

      const usize groups = deconvolution->groups;
      const usize output_size = output_height * output_width;
      const usize output_tile_size = pytorch_qnnp_params.q8conv.mr;
      const usize tiled_output_size = round_up(output_size, output_tile_size);
      const usize indirection_buffer_size =
          sizeof(void*) * batch_size * groups * tiled_output_size * kernel_size;

      const void** indirection_buffer = (const void**)realloc(
          deconvolution->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        pytorch_qnnp_log_error(
            "failed to allocate %zu bytes for indirection buffer",
            indirection_buffer_size);
        return pytorch_qnnp_status_out_of_memory;
      }
      deconvolution->indirection_buffer = indirection_buffer;

      pytorch_qnnp_indirection_init_deconv2d(
          deconvolution, output_tile_size, tiled_output_size);

      return pytorch_qnnp_status_success;
        */
}
