crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-run.cc]

pub struct Q8GemmContext {
    k:                   usize,
    k_stride:            usize,
    n:                   usize,
    n_stride:            usize,
    a:                   *const u8,
    a_stride:            usize,
    packed_w:            *const u8,
    c:                   *mut u8,
    c_stride:            usize,
    quantization_params: PyTorchQnnpConvQuantizationParams,
    ukernel:             PyTorchQ8GemmUKernelFunction,
}

pub fn compute_q8gemm(
    context:        [Q8GemmContext; 1],
    group_index:    usize,
    pixel_index:    usize,
    mr_block_start: usize,
    nr_block_start: usize,
    /* always 1 */
    group_range:    usize,
    pixel_range:    usize,
    mr_block_size:  usize,
    nr_block_size:  usize)  {
    
    todo!();
        /*
            const usize k = context->k;
      const usize k_stride = context->k_stride;
      const usize n = context->n;
      const usize n_stride = context->n_stride;
      const u8* a = context->a;
      const usize a_stride = context->a_stride;
      const void* packed_w = context->packed_w;
      u8* c = context->c;
      const usize c_stride = context->c_stride;

      usize output_channel_index = nr_block_start;
      context->ukernel(
          mr_block_size,
          nr_block_size,
          k,
          a + (pixel_index + mr_block_start) * a_stride + group_index * k,
          a_stride,
          (const void*) ((uintptr_t) packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(u8) + sizeof(i32))),
          c + (pixel_index + mr_block_start) * c_stride + nr_block_start + group_index * n,
          c_stride,
          output_channel_index,
          &context->quantization_params);
        */
}

pub fn qnnpack_linear(
    batch_size:            usize,
    input_channels:        usize,
    output_channels:       usize,
    input_zero_point:      u8,
    kernel_zero_points:    *const u8,
    requantization_scales: *const f32,
    output_zero_point:     u8,
    output_min:            u8,
    output_max:            u8,
    input:                 *const u8,
    input_stride:          usize,
    packed_weights:        *mut void,
    output:                *mut u8,
    output_stride:         usize,
    threadpool:            threadpool::ThreadPool) -> PyTorchQnnpStatus {

    todo!();
        /*
            const usize groups = 1;
      const usize group_input_channels = input_channels;
      const usize group_output_channels = output_channels;
      const u32 mr = pytorch_qnnp_params.q8conv.mr;
      const u32 nr = pytorch_qnnp_params.q8conv.nr;
      const u32 kr = pytorch_qnnp_params.q8conv.kr;
      const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
      const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

      const usize output_size = batch_size * 1;
      union pytorch_qnnp_conv_quantization_params conv_quantization_params =
          pytorch_qnnp_compute_conv_quantization_params(
              input_zero_point, kernel_zero_points,
              requantization_scales, output_zero_point, output_min, output_max);

      struct q8gemm_context q8gemm_context = {
          .k = group_input_channels,
          .k_stride = k_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .a = input,
          .a_stride = input_stride,
          .packed_w = (u8*) packed_weights,
          .c = output,
          .c_stride = output_stride,
          .quantization_params = conv_quantization_params,
          .ukernel = pytorch_qnnp_params.q8conv.gemm,
      };

      if (output_size == 0) {
          // pthreadpool can tolerate a range of 0, but not a tile of 0.
          // We use output_size as a tile size, so bail here if it's 0.
          return pytorch_qnnp_status_success;
      }

      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t) compute_q8gemm,
          &q8gemm_context,
          groups,
          1 * output_size,
          output_size,
          group_output_channels,
          1,
          output_size,
          mr,
          nr);

      return pytorch_qnnp_status_success;
        */
}
