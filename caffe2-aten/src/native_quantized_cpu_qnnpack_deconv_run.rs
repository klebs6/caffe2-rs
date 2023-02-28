crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/deconv-run.cc]

pub struct Q8ConvContext {
    bs:                  usize,
    ks:                  usize,
    kc:                  usize,
    kc_stride:           usize,
    m:                   usize,
    m_stride:            usize,
    n:                   usize,
    n_stride:            usize,
    indirect_a:          *const *const u8,
    packed_w:            *const void,
    c:                   *mut u8,
    c_stride:            usize,
    quantization_params: PyTorchQnnpConvQuantizationParams,
    ukernel:             PyTorchQ8ConvUKernelFunction,
}

pub fn compute_q8conv(
    context:        [Q8ConvContext; 1],
    group_index:    usize,
    image_index:    usize,
    mr_block_start: usize,
    nr_block_start: usize,

    /* always 1 */
    group_range:    usize,

    /* always 1 */
    image_range:    usize,
    mr_block_size:  usize,
    nr_block_size:  usize)  {

    todo!();
        /*
            const usize bs = context->bs;
      const usize ks = context->ks;
      const usize kc = context->kc;
      const usize kc_stride = context->kc_stride;
      const usize m = context->m;
      const usize m_stride = context->m_stride;
      const usize n = context->n;
      const usize n_stride = context->n_stride;
      const u8** indirect_a = context->indirect_a;
      const void* packed_w = context->packed_w;
      u8* c = context->c;
      const usize c_stride = context->c_stride;

      const usize output_channel_index = group_index * n + nr_block_start;
      context->ukernel(
          mr_block_size,
          nr_block_size,
          kc,
          ks,
          indirect_a +
              (mr_block_start + (image_index + group_index * bs) * m_stride) * ks,
          (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (kc_stride * sizeof(u8) + sizeof(i32))),
          c + (mr_block_start + image_index * m) * c_stride + group_index * n +
              nr_block_start,
          c_stride,
          output_channel_index,
          &context->quantization_params);
        */
}

pub struct QnnpackDeleter {

}

impl QnnpackDeleter {
    
    pub fn invoke(&mut self, op: PyTorchQnnpOperator)  {
        
        todo!();
        /*
            pytorch_qnnp_delete_operator(op);
        */
    }
}

pub fn qnnpack_de_conv(
    deconv_p:              &ConvParam,
    deconvolution:         PyTorchQnnpOperator,
    packed_weights:        *mut void,
    batch_size:            usize,
    input_height:          usize,
    input_width:           usize,
    input_zero_point:      u8,
    input:                 *const u8,
    kernel_zero_points:    *const u8,
    requantization_scales: *const f32,
    output_zero_point:     u8,
    output_min:            u8,
    output_max:            u8,
    output:                *mut u8,
    threadpool:            threadpool::ThreadPool) -> PyTorchQnnpStatus {

    todo!();
        /*
            if (batch_size == 0) {
        // Doesn't matter what's going on, if no batches, return
        return pytorch_qnnp_status_success;
      }
      // Check all invalid parameters
      const usize kernel_width = deconv_p.kernel_dims[0];
      const usize kernel_height = deconv_p.kernel_dims[1];

      // Support vars
      const usize group_input_channels = deconv_p.group_input_channels;
      const usize group_output_channels = deconv_p.group_output_channels;
      const u32 mr = pytorch_qnnp_params.q8conv.mr;
      const u32 nr = pytorch_qnnp_params.q8conv.nr;
      const u32 kr = pytorch_qnnp_params.q8conv.kr;
      const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
      const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

      // deconvolution->kernel_zero_point = deconv_p.kernel_zero_points;
      // const float kernel_scale = deconv_p.kernel_scale;
      // const float deconvolution_scale = input_scale * kernel_scale / output_scale;
      deconvolution->conv_quantization_params =
          pytorch_qnnp_compute_conv_quantization_params(
              input_zero_point,
              kernel_zero_points,
              requantization_scales,
              output_zero_point,
              output_min,
              output_max);

      // Setup the kernel
      const array<usize, 2> output_dims =
          deconv_p.compute_output_dims({input_width, input_height});
      const usize output_width = output_dims[0];
      const usize output_height = output_dims[1];
      const usize kernel_size = kernel_height * kernel_width;
      const usize output_size = output_height * output_width;

      const usize input_pixel_stride = deconv_p.input_channels;
      const usize output_pixel_stride = deconv_p.output_channels;

      if (deconvolution->input != input ||
          deconvolution->batch_size != batch_size ||
          deconvolution->input_height != input_height ||
          deconvolution->input_width != input_width ||
          deconvolution->input_pixel_stride != input_pixel_stride) {
        pytorch_qnnp_status status = pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
            deconvolution,
            batch_size,
            input_height,
            input_width,
            input,
            input_pixel_stride,
            output,
            output_pixel_stride,
            threadpool);
        if (status != pytorch_qnnp_status_success) {
          pytorch_qnnp_log_error(
              "failed to run decovolution op setup to setup indirection buffer.");
          return status;
        }
      }

      // Run the kernel
      const usize m_stride = round_up(output_size, mr);
      struct q8conv_context q8conv_context = {
          .bs = deconvolution->batch_size,
          .ks = kernel_size,
          .kc = group_input_channels,
          .kc_stride = k_stride * kernel_size,
          .m = output_size,
          .m_stride = m_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .indirect_a = (const u8**)deconvolution->indirection_buffer,
          .packed_w = packed_weights,
          .c = output,
          .c_stride = deconvolution->output_pixel_stride,
          .quantization_params = deconvolution->conv_quantization_params,
          .ukernel = pytorch_qnnp_params.q8conv.conv,
      };

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8conv,
          &q8conv_context,
          deconvolution->groups,
          batch_size,
          output_size,
          group_output_channels,
          1,
          1,
          mr,
          nr);
      return pytorch_qnnp_status_success;
        */
}
