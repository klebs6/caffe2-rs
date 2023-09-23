crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/conv-run.cc]

pub struct Q8GemmXzpContext {
    k:                     usize,
    k_stride:              usize,
    n:                     usize,
    n_stride:              usize,
    a:                     *const u8,
    a_stride:              usize,
    packed_w:              *const void,
    c:                     *mut u8,
    c_stride:              usize,
    a_sum:                 *const i32,
    groups:                usize,
    batch_size:            usize,
    a_sum_stride:          usize,
    requantization_params: PyTorchQnnpQ31RequantizationParams,
    ukernel:               PyTorchQ8GemmXzpUKernelFunction,
}

impl Q8GemmXzpContext {

    pub fn compute_q8gemm_xzp(
        context:        [Q8GemmXzpContext; 1],
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
      const i32* a_sum = context->a_sum;
      const usize groups = context->groups;
      const usize a_sum_stride = context->a_sum_stride;

      context->ukernel(
          mr_block_size,
          nr_block_size,
          k,
          a + (pixel_index + mr_block_start) * a_stride + group_index * k,
          a_stride,
          a_sum + pixel_index * groups + group_index * a_sum_stride +
              mr_block_start,
          (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(u8) + sizeof(i32))),
          c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
              group_index * n,
          c_stride,
          &context->requantization_params);
        */
    }
}

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

      const usize output_channel_index = nr_block_start + group_index * n;
      context->ukernel(
          mr_block_size,
          nr_block_size,
          k,
          a + (pixel_index + mr_block_start) * a_stride + group_index * k,
          a_stride,
          (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(u8) + sizeof(i32))),
          c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
              group_index * n,
          c_stride,
          output_channel_index,
          &context->quantization_params);
        */
}

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

pub struct Q8SumRowsContext {
    a:            *const u8,
    groups:       usize,
    m:            usize,
    k:            usize,
    a_stride:     usize,
    multiplier:   i32,
    a_sum:        *mut i32,
    a_sum_stride: usize,
    ukernel:      PyTorchQ8SumRowsUKernelFunction,
}

pub fn compute_sum_rows(
    context:     [Q8SumRowsContext; 1],
    group_index: usize,
    batch_index: usize,
    block_start: usize,

    /* always 1 */
    group_range: usize,

    /* always 1 */
    batch_range: usize,
    block_size:  usize)  {

    todo!();
        /*
            const u8* a = context->a;
      const usize groups = context->groups;
      const usize m = context->m;
      const usize k = context->k;
      const usize a_stride = context->a_stride;
      const i32 multiplier = context->multiplier;
      i32* a_sum = context->a_sum;
      const usize a_sum_stride = context->a_sum_stride;

      context->ukernel(
          a + batch_index * m * a_stride + group_index * k + block_start * a_stride,
          min(block_size, m - block_start),
          k,
          a_stride,
          multiplier,
          a_sum + batch_index * groups * a_sum_stride + group_index * a_sum_stride +
              block_start);
        */
}

pub struct Q8DwConvContext {
    groups:                        usize,
    group_stride:                  usize,
    indirection_buffer:            *const *const u8,
    indirection_buffer_row_stride: usize,
    indirection_buffer_col_stride: usize,
    packed_weights:                *const void,
    output:                        *mut u8,
    output_height:                 usize,
    output_width:                  usize,
    output_row_stride:             usize,
    output_col_increment:          usize,
    quantization_params:           PyTorchQnnpConvQuantizationParams,
    unipass_ukernel:               PyTorchQ8DwConvUpUKernelFunction,
    multipass_ukernel:             PyTorchQ8DwConvMpUKernelFunction,
}

pub fn compute_dwconv_unipass(
    context:  [Q8DwConvContext; 1],
    image:    usize,
    output_y: usize)  {
    
    todo!();
        /*
            const usize output_height = context->output_height;

      context->unipass_ukernel(
          context->groups,
          context->output_width,
          context->indirection_buffer +
              (image * output_height + output_y) *
                  context->indirection_buffer_row_stride,
          context->packed_weights,
          context->output +
              (image * output_height + output_y) * context->output_row_stride,
          context->indirection_buffer_col_stride,
          context->output_col_increment,
          &context->quantization_params);
        */
}

pub fn compute_dwconv_multiipass(
    context:  [Q8DwConvContext; 1],
    image:    usize,
    output_y: usize)  {
    
    todo!();
        /*
            const usize output_height = context->output_height;
      PYTORCH_QNNP_ALIGN(16)
    #ifdef _MSC_VER
      i32* multipass_acc = (i32*)_malloca(sizeof(i32) * context->group_stride);
    #else
      i32 multipass_acc[context->group_stride];
    #endif

      context->multipass_ukernel(
          context->groups,
          context->output_width,
          context->indirection_buffer +
              (image * output_height + output_y) *
                  context->indirection_buffer_row_stride,
          context->packed_weights,
          multipass_acc,
          context->output +
              (image * output_height + output_y) * context->output_row_stride,
          context->indirection_buffer_col_stride,
          context->output_col_increment,
          &context->quantization_params);

    #ifdef _MSC_VER
      _freea(multipass_acc);
    #endif
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

pub fn qnnpack_conv(
    conv_p:                &ConvParam,
    convolution:           PyTorchQnnpOperator,
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
            const usize input_pixel_stride = conv_p.input_channels;
      const usize output_pixel_stride = conv_p.output_channels;
      const usize kernel_width = conv_p.kernel_dims[0];
      const usize kernel_height = conv_p.kernel_dims[1];
      const usize kernel_size = kernel_height * kernel_width;
      const usize dilation_width = conv_p.dilation[0];
      const usize groups = conv_p.groups;

      if (batch_size == 0) {
        // If no batches, return
        return pytorch_qnnp_status_success;
      }

      union pytorch_qnnp_q31_requantization_params requantization_params;
      union pytorch_qnnp_conv_quantization_params conv_quantization_params;
      if (conv_p.ukernel_type == pytorch_qnnp_ukernel_type_xzp_gemm) {
        requantization_params = pytorch_qnnp_compute_requantization_params(
            // Note. XZP kernels are not changed for per channel quant.
            requantization_scales[0],
            output_zero_point,
            output_min,
            output_max);
      } else {
        conv_quantization_params = pytorch_qnnp_compute_conv_quantization_params(
            input_zero_point,
            kernel_zero_points,
            requantization_scales,
            output_zero_point,
            output_min,
            output_max);
      }
      u32 stride_width = conv_p.stride_dims[0];

      // Convolution op caches a few things.
      // We need to check if the corresponding values on this
      // invocation is same as cached values.
      // If so we can skip setup step.
      if (convolution->input != input ||
          convolution->batch_size != batch_size ||
          convolution->input_height != input_height ||
          convolution->input_width != input_width ||
          convolution->input_pixel_stride != input_pixel_stride) {
        pytorch_qnnp_status status = pytorch_qnnp_setup_convolution2d_nhwc_q8(
            convolution,
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
              "failed to run covolution op setup to setup indirection buffer.");
          return status;
        }
      }

      const usize output_size = convolution->output_height * convolution->output_width;

      switch (conv_p.ukernel_type) {
        case pytorch_qnnp_ukernel_type_dwconv: {
          const usize width_step =
              dilation_width == 1 ? stride_width : kernel_width;
          const u32 cr = pytorch_qnnp_params.q8dw9.cr;
          const usize group_stride = (groups + (cr - 1)) & -cr;

          switch (kernel_size) {
            case 9: {
              struct q8dwconv_context context = {
                  .groups = groups,
                  .group_stride = group_stride,
                  .indirection_buffer =
                    (const u8**)convolution->indirection_buffer,
                  .indirection_buffer_row_stride =
                      kernel_size +
                      (convolution->output_width * width_step - 1) * kernel_height,
                  .indirection_buffer_col_stride =
                      kernel_height * width_step * sizeof(void*),
                  .packed_weights = packed_weights,
                  .output = output,
                  .output_height = convolution->output_height,
                  .output_width = convolution->output_width,
                  .output_row_stride = convolution->output_width * output_pixel_stride,
                  .output_col_increment =
                      (output_pixel_stride - groups) * sizeof(u8),
                  .quantization_params = conv_quantization_params,
                  .unipass_ukernel =
                      conv_p.per_channel ?
                          pytorch_qnnp_params.q8dw9.updw_per_channel :
                          pytorch_qnnp_params.q8dw9.updw,
                  .multipass_ukernel =
                      conv_p.per_channel ?
                          pytorch_qnnp_params.q8dw25.mpdw_per_channel :
                          pytorch_qnnp_params.q8dw25.mpdw,
              };
              pthreadpool_compute_2d(
                  threadpool,
                  (pthreadpool_function_2d_t)compute_dwconv_unipass,
                  &context,
                  batch_size,
                  convolution->output_height);
              break;
            }
            case 25: {
              struct q8dwconv_context context = {
                  .groups = groups,
                  .group_stride = group_stride,
                  .indirection_buffer =
                      (const u8**)convolution->indirection_buffer,
                  .indirection_buffer_row_stride =
                      kernel_size +
                      (convolution->output_width * width_step - 1) * kernel_height,
                  .indirection_buffer_col_stride =
                      kernel_height * width_step * sizeof(void*),
                  .packed_weights = packed_weights,
                  .output = output,
                  .output_height = convolution->output_height,
                  .output_width = convolution->output_width,
                  .output_row_stride = convolution->output_width * output_pixel_stride,
                  .output_col_increment =
                      (output_pixel_stride - groups) * sizeof(u8),
                  .quantization_params = conv_quantization_params,
                  .unipass_ukernel =
                      conv_p.per_channel ?
                          pytorch_qnnp_params.q8dw9.updw_per_channel :
                          pytorch_qnnp_params.q8dw9.updw,
                  .multipass_ukernel =
                      conv_p.per_channel ?
                          pytorch_qnnp_params.q8dw25.mpdw_per_channel :
                          pytorch_qnnp_params.q8dw25.mpdw,
              };
              pthreadpool_compute_2d(
                  threadpool,
                  (pthreadpool_function_2d_t)compute_dwconv_multiipass,
                  &context,
                  batch_size,
                  convolution->output_height);
              break;
            }
            default:
              PYTORCH_QNNP_UNREACHABLE;
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_xzp_gemm: {
          const usize group_input_channels = conv_p.group_input_channels;
          const usize group_output_channels = conv_p.group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv_xzp.mr;
          const u32 nr = pytorch_qnnp_params.q8conv_xzp.nr;
          const u32 kr = pytorch_qnnp_params.q8conv_xzp.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

          /* compute input row sum */
          const usize input_size = input_height * input_width;
          i32* a_sum = (i32*)realloc(
              convolution->a_sum,
              sizeof(i32) * batch_size * groups * input_height * input_width);
          if (a_sum == nullptr) {
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for row sum data",
                sizeof(i32) * batch_size * groups * input_height * input_width);
            return pytorch_qnnp_status_out_of_memory;
          }
          convolution->a_sum = a_sum;
          struct q8sum_rows_context context = {
              .a = input,
              .groups = groups,
              .m = input_size,
              .k = conv_p.group_input_channels,
              .a_stride = input_pixel_stride,
              // XZP kernels are not supporting per channel quant.
              // We dont really use XZP kernels ATM.
              // Thus assigning the zero point of first channel.
              .multiplier = (i32)-kernel_zero_points[0],
              .a_sum = a_sum,
              .a_sum_stride = input_size,
              .ukernel = pytorch_qnnp_params.q8sum_rows.sum_rows,
          };
          pthreadpool_compute_3d_tiled(
              threadpool,
              (pthreadpool_function_3d_tiled_t)compute_sum_rows,
              &context,
              groups,
              batch_size,
              input_size,
              1,
              1,
              pytorch_qnnp_params.q8sum_rows.m);

          struct q8gemm_xzp_context q8gemm_xzp_context = {
              .k = conv_p.group_input_channels,
              .k_stride = k_stride,
              .n = conv_p.group_output_channels,
              .n_stride = n_stride,
              .a = input,
              .a_stride = input_pixel_stride,
              .packed_w = packed_weights,
              .c = output,
              .c_stride = output_pixel_stride,
              .a_sum = a_sum,
              .groups = groups,
              .batch_size = batch_size,
              .a_sum_stride = input_size,
              .requantization_params = requantization_params,
              .ukernel = pytorch_qnnp_params.q8conv_xzp.gemm,
          };
          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8gemm_xzp,
              &q8gemm_xzp_context,
              groups,
              batch_size * input_size,
              input_size,
              group_output_channels,
              1,
              input_size,
              mr,
              nr);
          break;
        }
        case pytorch_qnnp_ukernel_type_gemm: {
          const usize group_input_channels = conv_p.group_input_channels;
          const usize group_output_channels = conv_p.group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv.mr;
          const u32 nr = pytorch_qnnp_params.q8conv.nr;
          const u32 kr = pytorch_qnnp_params.q8conv.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

          struct q8gemm_context q8gemm_context = {
              .k = conv_p.group_input_channels,
              .k_stride = k_stride,
              .n = conv_p.group_output_channels,
              .n_stride = n_stride,
              .a = input,
              .a_stride = input_pixel_stride,
              .packed_w = (u8*)packed_weights,
              .c = output,
              .c_stride = output_pixel_stride,
              .quantization_params = conv_quantization_params,
              .ukernel = pytorch_qnnp_params.q8conv.gemm,
          };

          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8gemm,
              &q8gemm_context,
              groups,
              batch_size * output_size,
              output_size,
              group_output_channels,
              1,
              output_size,
              mr,
              nr);
          break;
        }
        case pytorch_qnnp_ukernel_type_conv: {
          const usize group_input_channels = conv_p.group_input_channels;
          const usize group_output_channels = conv_p.group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv.mr;
          const u32 nr = pytorch_qnnp_params.q8conv.nr;
          const u32 kr = pytorch_qnnp_params.q8conv.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;
          const usize m_stride = round_up(output_size, mr);

          struct q8conv_context q8conv_context = {
              .bs = batch_size,
              .ks = kernel_size,
              .kc = group_input_channels,
              .kc_stride = k_stride * kernel_size,
              .m = output_size,
              .m_stride = m_stride,
              .n = group_output_channels,
              .n_stride = n_stride,
              .indirect_a = (const u8**)convolution->indirection_buffer,
              .packed_w = packed_weights,
              .c = output,
              .c_stride = output_pixel_stride,
              .quantization_params = conv_quantization_params,
              .ukernel = pytorch_qnnp_params.q8conv.conv,
          };

          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8conv,
              &q8conv_context,
              groups,
              batch_size,
              output_size,
              group_output_channels,
              1,
              1,
              mr,
              nr);
          break;
        }
        default: {
          pytorch_qnnp_log_error("Invalid kernel type. QNNPACK convolution run failed.");
          PYTORCH_QNNP_UNREACHABLE;
        }
      }
      return pytorch_qnnp_status_success;
        */
}
