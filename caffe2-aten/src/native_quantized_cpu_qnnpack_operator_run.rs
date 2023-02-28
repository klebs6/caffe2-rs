crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/operator-run.c]

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
      const u8* restrict a = context->a;
      const usize a_stride = context->a_stride;
      const void* restrict packed_w = context->packed_w;
      u8* restrict c = context->c;
      const usize c_stride = context->c_stride;

      usize output_channel_index = nr_block_start + group_index * n;
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

/**
  | At the moment we opt to remove sparse
  | kernels that dont require prepacking
  | as their perf was always worse.
  |
  */
#[cfg(NO_PREPACK_SPARSE_KERNEL)]
pub struct Q8GemmSparseDqContext {
    a:                   *const u8,
    a_stride:            usize,
    kernel_col_indices:  *const u32,
    kernel_row_values:   *const u32,
    kernel_values:       *const u8,
    bias:                *const f32,

    /**
      | can be float or uint8)t
      |
      */
    c:                   *mut f32,

    c_stride:            usize,
    quantization_params: PyTorchQnnpConvDynamicQuantizationParams,
    ukernel:             PyTorchQ8GemmDqSparseUKernelFunction,
}

#[cfg(NO_PREPACK_SPARSE_KERNEL)]
pub fn compute_q8gemm_sparse_dq(
    context:        [Q8GemmSparseDqContext; 1],

    /* ignored */
    group_index:    usize,

    /* ignored */
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
            const u8* restrict a = context->a;
      const usize a_stride = context->a_stride;
      float* restrict c = (float*)context->c;
      const usize c_stride = context->c_stride;

      usize output_channel_index = nr_block_start;
      context->ukernel(
          mr_block_size,
          nr_block_size,
          a + mr_block_start * a_stride,
          a_stride,
          context->kernel_values,
          context->kernel_row_values + nr_block_start,
          context->kernel_col_indices,
          context->bias + nr_block_start,
          c + mr_block_start * c_stride + nr_block_start,
          c_stride,
          output_channel_index,
          &context->quantization_params);
        */
}

pub struct Q8GemmPrepackASparseDqContext {
    k:                   usize,
    a:                   *const u8,
    a_stride:            usize,
    a_packed:            *mut u8,
    a_packed_stride:     usize,
    log2_mr:             usize,
    log2_row_block_size: usize,
    kernel_col_indices:  *const u32,
    kernel_row_values:   *const u32,
    kernel_values:       *const u8,
    bias:                *const f32,

    /**
      | can be float or uint8)t
      |
      */
    c:                   *mut f32,

    c_stride:            usize,
    quantization_params: PyTorchQnnpConvDynamicQuantizationParams,
    ukernel:             PyTorchQ8GemmDqSparsePAckedAUKernelFunction,
    prepack_ukernel:     PyTorchQ8GemmSparsePackAUKernelFunction,
}

pub fn compute_q8gemm_prepack_a_sparse(
    context:        [Q8GemmPrepackASparseDqContext; 1],

    /* ignored */
    group_index:    usize,

    /* ignored */
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
            const u8* restrict a = context->a;
      const usize a_stride = context->a_stride;
      const usize mr_packed_block_start =
        ((mr_block_start >> context->log2_mr) * context->a_packed_stride);

      context->prepack_ukernel(
          mr_block_size,
          context->k,
          a + mr_block_start * a_stride,
          a_stride,
          context->a_packed + mr_packed_block_start);
        */
}

pub fn compute_q8gemm_prepacked_sparse_dq(
    context:        [Q8GemmPrepackASparseDqContext; 1],

    /* ignored */
    group_index:    usize,

    /* ignored */
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
            const u8* restrict a_packed = context->a_packed;
      const usize mr_packed_block_start =
        ((mr_block_start >> context->log2_mr) * context->a_packed_stride);
      float* restrict c = (float*)context->c;
      const usize c_stride = context->c_stride;

      usize output_channel_index = nr_block_start;
      context->ukernel(
          mr_block_size,
          nr_block_size,
          a_packed + mr_packed_block_start,
          context->kernel_values,
          context->kernel_row_values +
            (nr_block_start >> context->log2_row_block_size),
          context->kernel_col_indices,
          context->bias + nr_block_start,
          c + mr_block_start * c_stride + nr_block_start,
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
      const u8* restrict a = context->a;
      const usize a_stride = context->a_stride;
      const void* restrict packed_w = context->packed_w;
      u8* restrict c = context->c;
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
      const u8** restrict indirect_a = context->indirect_a;
      const void* restrict packed_w = context->packed_w;
      u8* restrict c = context->c;
      const usize c_stride = context->c_stride;

      usize output_channel_index = nr_block_start + group_index * n;
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

pub union Q8DwConvContextUnion {
    unipass_ukernel:   PyTorchQ8DwConvUpUKernelFunction,
    multipass_ukernel: PyTorchQ8DwConvMpUKernelFunction,
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
    u:                             Q8DwConvContextUnion,
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
      i32* multipass_acc = _malloca(sizeof(i32) * context->group_stride);
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

pub struct MaxPoolingContext {
    indirect_input:               *const *const void,
    indirect_input_batch_stride:  usize,
    indirect_input_height_stride: usize,
    output:                       *mut void,
    output_batch_stride:          usize,
    output_height_stride:         usize,
    output_width:                 usize,
    pooling_size:                 usize,
    channels:                     usize,
    input_increment:              usize,
    output_increment:             usize,
    params:                       PyTorchQnnpU8ClampingParams,
    ukernel:                      PyTorchU8MaxPoolUKernelFunction,
}

pub fn compute_max_pooling(
    context:     [MaxPoolingContext; 1],
    batch_index: usize,
    output_y:    usize)  {
    
    todo!();
        /*
            const void** indirect_input =
        (const void**) ((uintptr_t) context->indirect_input +
          batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
      void* output =
        (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);

      context->ukernel(
          context->output_width,
          context->pooling_size,
          context->channels,
          (const u8**)indirect_input,
          output,
          context->input_increment,
          context->output_increment,
          &context->params);
        */
}

pub union AveragePoolingContextUnion {
    unipass_ukernel:   PyTorchQ8AvgPoolUpUKernelFunction,
    multipass_ukernel: PyTorchQ8AvgPoolMpUKernelFunction,
}

pub struct AveragePoolingContext {
    indirect_input:               *const *const void,
    indirect_input_batch_stride:  usize,
    indirect_input_height_stride: usize,
    output:                       *mut void,
    output_batch_stride:          usize,
    output_height_stride:         usize,
    output_width:                 usize,
    pooling_size:                 usize,
    channels:                     usize,
    packed_channels:              usize,
    zero:                         *const void,
    input_increment:              usize,
    output_increment:             usize,
    quantization_params:          PyTorchQnnpAvgPoolQuantizationParams,
    u:                            AveragePoolingContextUnion,
}

pub fn compute_average_pooling_unipass(
    context:     [AveragePoolingContext; 1],
    batch_index: usize,
    output_y:    usize)  {
    
    todo!();
        /*
            const void** indirect_input =
        (const void**) ((uintptr_t) context->indirect_input +
          batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
      void* output =
        (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);

      context->unipass_ukernel(
          context->output_width,
          context->pooling_size,
          context->channels,
          (const u8**)indirect_input,
          context->zero,
          output,
          context->input_increment,
          context->output_increment,
          &context->quantization_params);
        */
}

pub fn compute_average_pooling_multipass(
    context:     [AveragePoolingContext; 1],
    batch_index: usize,
    output_y:    usize)  {
    
    todo!();
        /*
            const void** indirect_input =
        (const void**) ((uintptr_t) context->indirect_input +
          batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
      void* output =
        (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);
      PYTORCH_QNNP_ALIGN(16)
    #ifdef _MSC_VER
      i32* multipass_buffer =
          _malloca(sizeof(i32) * context->packed_channels);
    #else
      i32 multipass_buffer[context->packed_channels];
    #endif

      context->multipass_ukernel(
          context->output_width,
          context->pooling_size,
          context->channels,
          (const u8**)indirect_input,
          context->zero,
          multipass_buffer,
          output,
          context->input_increment,
          context->output_increment,
          &context->quantization_params);

    #ifdef _MSC_VER
      _freea(multipass_buffer);
    #endif
        */
}

pub union GlobalAveragePoolingContextUnion {
    unipass_ukernel:   PyTorchQ8GAvgPoolUpUKernelFunction,
    multipass_ukernel: PyTorchQ8GAvgPoolMpUKernelFunction,
}

pub struct GlobalAveragePoolingContext {
    input:               *const void,
    zero:                *const void,
    input_pixel_stride:  usize,
    input_batch_stride:  usize,
    input_elements:      usize,
    channels:            usize,
    packed_channels:     usize,
    output:              *mut void,
    output_batch_stride: usize,
    quantization_params: PyTorchQnnpAvgPoolQuantizationParams,
    u:                   GlobalAveragePoolingContextUnion,
}

pub fn compute_global_average_pooling_unipass(
    context:     [GlobalAveragePoolingContext; 1],
    batch_index: usize)  {
    
    todo!();
        /*
            const void* input =
          (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
      void* output =
          (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);

      context->unipass_ukernel(
          context->input_elements,
          context->channels,
          input,
          context->input_pixel_stride,
          context->zero,
          output,
          &context->quantization_params);
        */
}

pub fn compute_global_average_pooling_multipass(
    context:     [GlobalAveragePoolingContext; 1],
    batch_index: usize)  {
    
    todo!();
        /*
            const void* input =
          (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
      void* output =
          (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);
      PYTORCH_QNNP_ALIGN(16)
    #ifdef _MSC_VER
      i32* multipass_buffer =
          _malloca(sizeof(i32) * context->packed_channels);
    #else
      i32 multipass_buffer[context->packed_channels];
    #endif

      context->multipass_ukernel(
          context->input_elements,
          context->channels,
          input,
          context->input_pixel_stride,
          context->zero,
          multipass_buffer,
          output,
          &context->quantization_params);

    #ifdef _MSC_VER
      _freea(multipass_buffer);
    #endif
        */
}

pub struct Q8AddStridedContext {
    n:                   usize,
    a:                   *const u8,
    a_stride:            usize,
    b:                   *const u8,
    b_stride:            usize,
    y:                   *const u8,
    y_stride:            usize,
    quantization_params: PyTorchQnnpAddQuantizationParams,
    ukernel:             PyTorchQ8VAddUKernelFunction,
}

pub fn compute_q8add_strided(
    context:      [Q8AddStridedContext; 1],
    batch_offset: usize,

    /* always 1 */
    batch_range:  usize)
{
    todo!();
        /*
            assert(batch_range == 1);

      const usize n = context->n;
      const usize a_stride = context->a_stride;
      const usize b_stride = context->b_stride;
      const usize y_stride = context->y_stride;
      const void* a =
          (const void*)((uintptr_t)context->a + a_stride * batch_offset);
      const void* b =
          (const void*)((uintptr_t)context->b + b_stride * batch_offset);
      void* y = (void*)((uintptr_t)context->y + y_stride * batch_offset);

      context->ukernel(n, a, b, y, &context->quantization_params);
        */
}

pub struct Q8AddContiguousContext {
    a:                   *const u8,
    b:                   *const u8,
    y:                   *mut u8,
    quantization_params: PyTorchQnnpAddQuantizationParams,
    ukernel:             PyTorchQ8VAddUKernelFunction,
}

pub fn compute_q8add_contiguous(
    context: [Q8AddContiguousContext; 1],
    offset:  usize,
    size:    usize)  {
    
    todo!();
        /*
            const void* a = (const void*)((uintptr_t)context->a + offset);
      const void* b = (const void*)((uintptr_t)context->b + offset);
      void* y = (void*)((uintptr_t)context->y + offset);
      context->ukernel(size, a, b, y, &context->quantization_params);
        */
}

pub union ChannelShuffleContextUnion {
    fixed_ukernel:    PyTorchXZipcUKernelFunction,
    variable_ukernel: PyTorchXZipvUKernelFunction,
}

pub struct ChannelShuffleContext {
    x:        *const void,
    x_stride: usize,
    y:        *mut void,
    y_stride: usize,
    n:        usize,
    m:        usize,
    u:        ChannelShuffleContextUnion,
}

pub fn compute_channel_shuffle_fixed(
        context: [ChannelShuffleContext; 1],
        index:   usize)  {
    
    todo!();
        /*
            const void* x =
          (const void*)((uintptr_t)context->x + index * context->x_stride);
      void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

      context->fixed_ukernel(context->n, x, y);
        */
}

pub fn compute_channel_shuffle_variable(
    context: [ChannelShuffleContext; 1],
    index:   usize)  {
    
    todo!();
        /*
            const void* x =
          (const void*)((uintptr_t)context->x + index * context->x_stride);
      void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

      context->variable_ukernel(context->n, context->m, x, y);
        */
}

pub struct LutStridedContext {
    n:        usize,
    x:        *const void,
    x_stride: usize,
    t:        *const void,
    y:        *mut void,
    y_stride: usize,
    ukernel:  PyTorchX8LutUKernelFunction,
}

pub fn compute_lut_strided(
    context:     [LutStridedContext; 1],
    batch_index: usize)  {

    todo!();
        /*
            const void* x =
          (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
      void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);

      context->ukernel(context->n, x, context->t, y);
        */
}

pub struct LutContiguousContext {
    x:        *const void,
    x_stride: usize,
    t:        *const void,
    y:        *mut void,
    y_stride: usize,
    ukernel:  PyTorchX8LutUKernelFunction,
}

pub fn compute_lut_contiguous(
    context: [LutContiguousContext; 1],
    offset:  usize,
    size:    usize)  {
    
    todo!();
        /*
            const void* x = (const void*)((uintptr_t)context->x + offset);
      void* y = (void*)((uintptr_t)context->y + offset);

      context->ukernel(size, x, context->t, y);
        */
}

pub struct ClampStridedContext {
    n:        usize,
    x:        *const void,
    x_stride: usize,
    y:        *mut void,
    y_stride: usize,
    ukernel:  PyTorchU8ClampUKernelFunction,
    params:   PyTorchQnnpU8ClampingParams,
}

pub fn compute_clamp_strided(
    context:     [ClampStridedContext; 1],
    batch_index: usize)  {
    
    todo!();
        /*
            const void* x =
          (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
      void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);
      context->ukernel(context->n, x, y, &context->params);
        */
}

pub struct ClampContiguousContext {
    x:        *const void,
    x_stride: usize,
    y:        *mut void,
    y_stride: usize,
    ukernel:  PyTorchU8ClampUKernelFunction,
    params:   PyTorchQnnpU8ClampingParams,
}

pub fn compute_clamp_contiguous(
    context: [ClampContiguousContext; 1],
    offset:  usize,
    size:    usize)  {
    
    todo!();
        /*
            const void* x = (const void*)((uintptr_t)context->x + offset);
      void* y = (void*)((uintptr_t)context->y + offset);
      context->ukernel(size, x, y, &context->params);
        */
}

pub struct U8SoftArgmaxContext {
    n:                usize,
    x:                *const u8,
    x_stride:         usize,
    t:                *const u32,
    y:                *mut u8,
    y_stride:         usize,
    rmax_ukernel:     PyTorchU8RMaxUKernelFunction,
    lut_norm_ukernel: PyTorchU8Lut32NormUKernelFunction,
}

pub fn compute_u8softargmax(
    context:     [U8SoftArgmaxContext; 1],
    batch_index: usize)  {
    
    todo!();
        /*
            const u8* x =
          (const u8*)((uintptr_t)context->x + context->x_stride * batch_index);
      u8* y =
          (u8*)((uintptr_t)context->y + context->y_stride * batch_index);
      const usize n = context->n;

      const u8 x_max = context->rmax_ukernel(n, x);
      const usize adjustment = x_max ^ 255;
      const u32* t = (const u32*)context->t + adjustment;
      context->lut_norm_ukernel(n, x, t, y);
        */
}

pub fn pytorch_qnnp_run_operator(
    op:         PyTorchQnnpOperator,
    threadpool: threadpool::ThreadPool) -> PyTorchQnnpStatus {

    todo!();
        /*
            // For any ukernel type, there is no work to do if the batch size is 0.
      if (op->batch_size == 0) {
        return pytorch_qnnp_status_success;
      }

      switch (op->ukernel_type) {
        case pytorch_qnnp_ukernel_type_dwconv: {
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize kernel_height = op->kernel_height;
          const usize kernel_width = op->kernel_width;
          const usize kernel_size = kernel_height * kernel_width;
          const usize width_step =
              op->dilation_width == 1 ? op->stride_width : op->kernel_width;
          const usize output_height = op->output_height;
          const usize output_width = op->output_width;

          switch (kernel_size) {
            case 9: {
              struct q8dwconv_context context = {
                  .groups = groups,
                  .indirection_buffer = (const u8**)op->indirection_buffer,
                  .indirection_buffer_row_stride =
                      kernel_size + (output_width * width_step - 1) * kernel_height,
                  .indirection_buffer_col_stride =
                      kernel_height * width_step * sizeof(void*),
                  .packed_weights = op->packed_weights,
                  .output = op->output,
                  .output_height = output_height,
                  .output_width = output_width,
                  .output_row_stride = output_width * op->output_pixel_stride,
                  .output_col_increment =
                      (op->output_pixel_stride - groups) * sizeof(u8),
                  .quantization_params = op->conv_quantization_params,
                  .unipass_ukernel =
                      op->per_channel ?
                          pytorch_qnnp_params.q8dw9.updw_per_channel :
                          pytorch_qnnp_params.q8dw9.updw,
              };
              pthreadpool_compute_2d(
                  threadpool,
                  (pthreadpool_function_2d_t)compute_dwconv_unipass,
                  &context,
                  batch_size,
                  output_height);
              break;
            }
            case 25: {
              struct q8dwconv_context context = {
                  .groups = groups,
                  .group_stride = op->group_stride,
                  .indirection_buffer = (const u8**)op->indirection_buffer,
                  .indirection_buffer_row_stride =
                      kernel_size + (output_width * width_step - 1) * kernel_height,
                  .indirection_buffer_col_stride =
                      kernel_height * width_step * sizeof(void*),
                  .packed_weights = op->packed_weights,
                  .output = op->output,
                  .output_height = output_height,
                  .output_width = output_width,
                  .output_row_stride = output_width * op->output_pixel_stride,
                  .output_col_increment =
                      (op->output_pixel_stride - groups) * sizeof(u8),
                  .quantization_params = op->conv_quantization_params,
                  .multipass_ukernel =
                      op->per_channel ?
                          pytorch_qnnp_params.q8dw25.mpdw_per_channel :
                          pytorch_qnnp_params.q8dw25.mpdw,
              };
              pthreadpool_compute_2d(
                  threadpool,
                  (pthreadpool_function_2d_t)compute_dwconv_multiipass,
                  &context,
                  batch_size,
                  output_height);
              break;
            }
            default:
              PYTORCH_QNNP_UNREACHABLE;
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_xzp_gemm: {
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize group_input_channels = op->group_input_channels;
          const usize group_output_channels = op->group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv_xzp.mr;
          const u32 nr = pytorch_qnnp_params.q8conv_xzp.nr;
          const u32 kr = pytorch_qnnp_params.q8conv_xzp.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

          /* compute input row sum */
          const usize input_size = op->input_height * op->input_width;
          i32* a_sum = (i32*)op->a_sum;

          struct q8sum_rows_context context = {
              .a = op->input,
              .groups = groups,
              .m = input_size,
              .k = group_input_channels,
              .a_stride = op->input_pixel_stride,
              .multiplier = (i32)-op->kernel_zero_point,
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
              .k = group_input_channels,
              .k_stride = k_stride,
              .n = group_output_channels,
              .n_stride = n_stride,
              .a = op->input,
              .a_stride = op->input_pixel_stride,
              .packed_w = op->packed_weights,
              .c = op->output,
              .c_stride = op->output_pixel_stride,
              .a_sum = a_sum,
              .groups = op->groups,
              .batch_size = batch_size,
              .a_sum_stride = input_size,
              .requantization_params = op->requantization_params,
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
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize group_input_channels = op->group_input_channels;
          const usize group_output_channels = op->group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv.mr;
          const u32 nr = pytorch_qnnp_params.q8conv.nr;
          const u32 kr = pytorch_qnnp_params.q8conv.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

          const usize output_size = op->output_height * op->output_width;
          struct q8gemm_context q8gemm_context = {
              .k = group_input_channels,
              .k_stride = k_stride,
              .n = group_output_channels,
              .n_stride = n_stride,
              .a = op->input,
              .a_stride = op->input_pixel_stride,
              .packed_w = op->packed_weights,
              .c = op->output,
              .c_stride = op->output_pixel_stride,
              .quantization_params = op->conv_quantization_params,
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
    #ifdef NO_PREPACK_SPARSE_KERNEL
        case pytorch_qnnp_ukernel_type_gemm_sparse_dq: {
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize group_output_channels = op->group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8gemm_sparse_c1x4.mr;
          const u32 nr = pytorch_qnnp_params.q8gemm_sparse_c1x4.nr;

          const usize output_size = op->output_height * op->output_width;
          struct q8gemm_sparse_dq_context q8gemm_sparse_dq_context = {
              .a = op->input,
              .a_stride = op->input_pixel_stride,
              .kernel_col_indices = op->sparse_matrix.col_indices,
              .kernel_row_values = op->sparse_matrix.row_values,
              .kernel_values = op->sparse_matrix.values,
              .bias = (const float*)op->bias,
              .c = (float*)op->output,
              .c_stride = op->output_pixel_stride,
              .quantization_params = op->dynamic_conv_quantization_params,
              .ukernel = pytorch_qnnp_params.q8gemm_sparse_c1x4.gemm_dq,
          };

          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8gemm_sparse_dq,
              &q8gemm_sparse_dq_context,
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
    #endif
        case pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq: {
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize group_input_channels = op->group_input_channels;
          const usize group_output_channels = op->group_output_channels;
          u32 mr, log2_mr, nr, kr, log2_row_block_size;
          pytorch_q8gemm_sparse_packA_ukernel_function prepack_kernel;
          pytorch_q8gemm_dq_sparse_packedA_ukernel_function compute_kernel;
          if (op->sparse_matrix.row_block_size == 1 &&
              op->sparse_matrix.col_block_size == 4) {
            mr = pytorch_qnnp_params.q8gemm_sparse_c1x4.mr;
            log2_mr = pytorch_qnnp_params.q8gemm_sparse_c1x4.log2_mr;
            log2_row_block_size = 0;
            nr = pytorch_qnnp_params.q8gemm_sparse_c1x4.nr;
            kr = pytorch_qnnp_params.q8gemm_sparse_c1x4.kr;
            compute_kernel =
              pytorch_qnnp_params.q8gemm_sparse_c1x4.packedA_gemm_dq;
            prepack_kernel = pytorch_qnnp_params.q8gemm_sparse_c1x4.packA;
          } else if (op->sparse_matrix.row_block_size == 8 &&
              op->sparse_matrix.col_block_size == 1) {
            mr = pytorch_qnnp_params.q8gemm_sparse_c8x1.mr;
            log2_mr = pytorch_qnnp_params.q8gemm_sparse_c8x1.log2_mr;
            log2_row_block_size = 3;
            nr = pytorch_qnnp_params.q8gemm_sparse_c8x1.nr;
            kr = pytorch_qnnp_params.q8gemm_sparse_c8x1.kr;
            compute_kernel =
              pytorch_qnnp_params.q8gemm_sparse_c8x1.packedA_gemm_dq;
            prepack_kernel = pytorch_qnnp_params.q8gemm_sparse_c8x1.packA;
          } else {
            return pytorch_qnnp_status_invalid_parameter;
          }

          const usize output_size = op->output_height * op->output_width;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize m_stride = (output_size + (mr - 1)) & -mr;
          op->prepacked_a =
            (u8*)realloc((void*)op->prepacked_a, k_stride * m_stride);
          if (op->prepacked_a == NULL) {
            pytorch_qnnp_log_error(
                "failed to allocate %zu bytes for packed activation buffer",
                (k_stride * m_stride));
            return pytorch_qnnp_status_out_of_memory;
          }

          struct q8gemm_prepackA_sparse_dq_context
            q8gemm_prepack_sparse_dq_context = {
              .k = group_input_channels,
              .a = op->input,
              .a_stride = op->input_pixel_stride,
              .a_packed = op->prepacked_a,
              .a_packed_stride = k_stride * mr,
              .log2_mr = log2_mr,
              .log2_row_block_size = log2_row_block_size,
              .kernel_col_indices = op->sparse_matrix.col_indices,
              .kernel_row_values = op->sparse_matrix.row_values,
              .kernel_values = op->sparse_matrix.values,
              .bias = (const float*)op->bias,
              .c = (float*)op->output,
              .c_stride = op->output_pixel_stride,
              .quantization_params = op->dynamic_conv_quantization_params,
              .ukernel = compute_kernel,
              .prepack_ukernel = prepack_kernel,
          };

          // This batch size is not the actual batch size of the op
          // The batch size is modified in fully-connected-sparse.c
          if (groups != 1 || batch_size != 1) {
            pytorch_qnnp_log_error("pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq "
                "works with group size = 1, batch_size = 1.\n");
            return pytorch_qnnp_status_invalid_parameter;
          }

          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8gemm_prepack_a_sparse,
              &q8gemm_prepack_sparse_dq_context,
              1,
              1,
              output_size,
              1,
              1,
              1,
              mr,
              1);

          pthreadpool_compute_4d_tiled(
              threadpool,
              (pthreadpool_function_4d_tiled_t)compute_q8gemm_prepacked_sparse_dq,
              &q8gemm_prepack_sparse_dq_context,
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
          const usize batch_size = op->batch_size;
          const usize groups = op->groups;
          const usize group_input_channels = op->group_input_channels;
          const usize group_output_channels = op->group_output_channels;
          const u32 mr = pytorch_qnnp_params.q8conv.mr;
          const u32 nr = pytorch_qnnp_params.q8conv.nr;
          const u32 kr = pytorch_qnnp_params.q8conv.kr;
          const usize k_stride = (group_input_channels + (kr - 1)) & -kr;
          const usize n_stride = (group_output_channels + (nr - 1)) & -nr;

          const usize output_size = op->output_height * op->output_width;
          const usize kernel_size = op->kernel_height * op->kernel_width;
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
              .indirect_a = (const u8**)op->indirection_buffer,
              .packed_w = op->packed_weights,
              .c = op->output,
              .c_stride = op->output_pixel_stride,
              .quantization_params = op->conv_quantization_params,
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
        case pytorch_qnnp_ukernel_type_average_pooling: {
          const u32 kr = pytorch_qnnp_params.q8avgpool.kr;
          const u32 mr = pytorch_qnnp_params.q8avgpool.mr;
          const u32 qr = pytorch_qnnp_params.q8avgpool.qr;
          const usize channels = op->channels;
          const usize output_width = op->output_width;
          const usize output_height = op->output_height;
          const usize pooling_height = op->kernel_height;
          const usize pooling_width = op->kernel_width;
          const usize pooling_size = pooling_height * pooling_width;

          const usize width_step = min(op->stride_width, pooling_width);
          const usize indirect_input_height_stride =
              (pooling_size + (output_width * width_step - 1) * pooling_height) *
              sizeof(void*);
          const usize output_height_stride =
              output_width * op->output_pixel_stride;

          usize multipass_adjustment = 0;
          if (channels >= kr && pooling_size > mr) {
            multipass_adjustment = round_up(pooling_size - mr, qr) + mr - qr;
          }
          struct average_pooling_context context = {
              .indirect_input = op->indirection_buffer,
              .indirect_input_batch_stride =
                  output_height * indirect_input_height_stride,
              .indirect_input_height_stride = indirect_input_height_stride,
              .output = op->output,
              .output_batch_stride = output_height * output_height_stride,
              .output_height_stride = output_height_stride,
              .output_width = output_width,
              .pooling_size = pooling_size,
              .channels = channels,
              .packed_channels = (channels + (kr - 1)) & -kr,
              .zero = op->zero_pointer,
              .input_increment =
                  (pooling_height * width_step - multipass_adjustment) *
                  sizeof(void*),
              .output_increment =
                  (op->output_pixel_stride - channels) * sizeof(u8),
              .quantization_params = op->avgpool_quantization_params,
          };

          pthreadpool_function_2d_t compute_function = NULL;
          if (channels < kr) {
            compute_function =
                (pthreadpool_function_2d_t)compute_average_pooling_unipass;
            context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.ltkr;
          } else {
            if (pooling_size <= mr) {
              compute_function =
                  (pthreadpool_function_2d_t)compute_average_pooling_unipass;
              context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_lemr;
            } else {
              compute_function =
                  (pthreadpool_function_2d_t)compute_average_pooling_multipass;
              context.multipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_gtmr;
            }
          }

          pthreadpool_compute_2d(
              threadpool,
              compute_function,
              &context,
              op->batch_size,
              output_height);
          break;
        }
        case pytorch_qnnp_ukernel_type_max_pooling: {
          const u32 kr = pytorch_qnnp_params.u8maxpool.kr;
          const u32 mr = pytorch_qnnp_params.u8maxpool.mr;
          const u32 qr = pytorch_qnnp_params.u8maxpool.qr;
          const usize channels = op->channels;
          const usize output_width = op->output_width;
          const usize output_height = op->output_height;
          const usize pooling_height = op->kernel_height;
          const usize pooling_width = op->kernel_width;
          const usize pooling_size = pooling_height * pooling_width;

          const usize width_step = op->dilation_width > 1
              ? pooling_width
              : min(op->stride_width, pooling_width);
          const usize indirect_input_height_stride =
              (pooling_size + (output_width * width_step - 1) * pooling_height) *
              sizeof(void*);
          const usize output_height_stride =
              output_width * op->output_pixel_stride;

          usize multipass_adjustment = pooling_size;
          if (channels >= kr) {
            multipass_adjustment = round_up(doz(pooling_size, mr), qr) + mr;
          }
          struct max_pooling_context context = {
              .indirect_input = op->indirection_buffer,
              .indirect_input_batch_stride =
                  output_height * indirect_input_height_stride,
              .indirect_input_height_stride = indirect_input_height_stride,
              .output = op->output,
              .output_batch_stride = output_height * output_height_stride,
              .output_height_stride = output_height_stride,
              .output_width = output_width,
              .pooling_size = pooling_size,
              .channels = channels,
              .input_increment =
                  (pooling_height * width_step - multipass_adjustment) *
                  sizeof(void*),
              .output_increment =
                  (op->output_pixel_stride - channels) * sizeof(u8),
              .params = op->u8_clamping_params,
              .ukernel = channels < kr ? pytorch_qnnp_params.u8maxpool.ltkr
                                       : pytorch_qnnp_params.u8maxpool.gekr,
          };

          pthreadpool_compute_2d(
              threadpool,
              (pthreadpool_function_2d_t)compute_max_pooling,
              &context,
              op->batch_size,
              output_height);
          break;
        };
        case pytorch_qnnp_ukernel_type_add: {
          const usize batch_size = op->batch_size;
          const usize channels = op->channels;
          const usize a_stride = op->input_pixel_stride;
          const usize b_stride = op->input2_pixel_stride;
          const usize y_stride = op->output_pixel_stride;
          if ((((a_stride ^ channels) | (b_stride ^ channels) |
                (y_stride ^ channels)) == 0) ||
              batch_size == 1) {
            const usize block_size = 4096;
            struct q8add_contiguous_context add_context = {
                .a = op->input,
                .b = op->input2,
                .y = op->output,
                .quantization_params = op->add_quantization_params,
                .ukernel = pytorch_qnnp_params.q8vadd,
            };
            pthreadpool_compute_1d_tiled(
                threadpool,
                (pthreadpool_function_1d_tiled_t)compute_q8add_contiguous,
                &add_context,
                batch_size * channels * sizeof(u8),
                block_size);
          } else {
            struct q8add_strided_context add_context = {
                .a = op->input,
                .a_stride = a_stride * sizeof(u8),
                .b = op->input2,
                .b_stride = b_stride * sizeof(u8),
                .y = op->output,
                .y_stride = y_stride * sizeof(u8),
                .n = channels,
                .quantization_params = op->add_quantization_params,
                .ukernel = pytorch_qnnp_params.q8vadd,
            };
            pthreadpool_compute_1d_tiled(
                threadpool,
                (pthreadpool_function_1d_tiled_t)compute_q8add_strided,
                &add_context,
                batch_size,
                1);
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_global_average_pooling: {
          const u32 nr = pytorch_qnnp_params.q8gavgpool.nr;
          const u32 mr = pytorch_qnnp_params.q8gavgpool.mr;
          const usize input_pixel_stride =
              op->input_pixel_stride * sizeof(u8);
          const usize input_width = op->input_width;
          const usize channels = op->channels;
          struct global_average_pooling_context context = {
              .input = op->input,
              .zero = op->zero_pointer,
              .input_pixel_stride = input_pixel_stride,
              .input_batch_stride = input_pixel_stride * input_width,
              .input_elements = input_width,
              .channels = channels,
              .packed_channels = (channels + (nr - 1)) & -nr,
              .output = op->output,
              .output_batch_stride = op->output_pixel_stride * sizeof(u8),
              .quantization_params = op->avgpool_quantization_params,
          };
          pthreadpool_function_1d_t compute_function = NULL;
          if (channels < nr) {
            compute_function =
                (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
            context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.ltnr;
          } else {
            if (input_width <= mr) {
              compute_function =
                  (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
              context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_lemr;
            } else {
              compute_function = (pthreadpool_function_1d_t)
                  compute_global_average_pooling_multipass;
              context.multipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_gtmr;
            }
          }

          pthreadpool_compute_1d(
              threadpool, compute_function, &context, op->batch_size);
          break;
        }
        case pytorch_qnnp_ukernel_type_lut: {
          const usize batch_size = op->batch_size;
          const usize channels = op->channels;
          const usize x_stride = op->input_pixel_stride;
          const usize y_stride = op->output_pixel_stride;
          if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
              batch_size == 1) {
            const usize block_size = 1024;
            struct lut_contiguous_context context = {
                .x = op->input,
                .x_stride = x_stride * sizeof(u8),
                .t = op->lookup_table,
                .y = op->output,
                .y_stride = y_stride * sizeof(u8),
                .ukernel = pytorch_qnnp_params.x8lut,
            };
            pthreadpool_compute_1d_tiled(
                threadpool,
                (pthreadpool_function_1d_tiled_t)compute_lut_contiguous,
                &context,
                batch_size * channels * sizeof(u8),
                block_size);
          } else {
            struct lut_strided_context context = {
                .n = channels,
                .x = op->input,
                .x_stride = x_stride * sizeof(u8),
                .t = op->lookup_table,
                .y = op->output,
                .y_stride = y_stride * sizeof(u8),
                .ukernel = pytorch_qnnp_params.x8lut,
            };
            pthreadpool_compute_1d(
                threadpool,
                (pthreadpool_function_1d_t)compute_lut_strided,
                &context,
                batch_size);
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_clamp: {
          const usize batch_size = op->batch_size;
          const usize channels = op->channels;
          const usize x_stride = op->input_pixel_stride;
          const usize y_stride = op->output_pixel_stride;
          if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
              batch_size == 1) {
            const usize block_size = 4096;
            struct clamp_contiguous_context context = {
                .x = op->input,
                .x_stride = x_stride * sizeof(u8),
                .y = op->output,
                .y_stride = y_stride * sizeof(u8),
                .ukernel = pytorch_qnnp_params.u8clamp,
                .params = op->u8_clamping_params,
            };
            pthreadpool_compute_1d_tiled(
                threadpool,
                (pthreadpool_function_1d_tiled_t)compute_clamp_contiguous,
                &context,
                batch_size * channels * sizeof(u8),
                block_size);
          } else {
            struct clamp_strided_context context = {
                .n = channels,
                .x = op->input,
                .x_stride = x_stride * sizeof(u8),
                .y = op->output,
                .y_stride = y_stride * sizeof(u8),
                .ukernel = pytorch_qnnp_params.u8clamp,
                .params = op->u8_clamping_params,
            };
            pthreadpool_compute_1d(
                threadpool,
                (pthreadpool_function_1d_t)compute_clamp_strided,
                &context,
                batch_size);
          }
          break;
        }
        case pytorch_qnnp_ukernel_type_softargmax: {
          struct u8softargmax_context context = {
              .n = op->channels,
              .x = op->input,
              .x_stride = op->input_pixel_stride * sizeof(u8),
              .t = op->lookup_table,
              .y = op->output,
              .y_stride = op->output_pixel_stride * sizeof(u8),
              .rmax_ukernel = pytorch_qnnp_params.u8rmax,
              .lut_norm_ukernel = pytorch_qnnp_params.u8lut32norm,
          };
          pthreadpool_compute_1d(
              threadpool,
              (pthreadpool_function_1d_t)compute_u8softargmax,
              &context,
              op->batch_size);
          break;
        }
        case pytorch_qnnp_ukernel_type_channel_shuffle: {
          const usize groups = op->groups;
          struct channel_shuffle_context channel_shuffle_context = {
              .x = op->input,
              .x_stride = op->input_pixel_stride * sizeof(u8),
              .y = op->output,
              .y_stride = op->output_pixel_stride * sizeof(u8),
              .n = op->group_channels * sizeof(u8),
              .m = groups,
          };
          pthreadpool_function_1d_t compute_function = NULL;
          switch (groups) {
            case 2:
              compute_function =
                  (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
              channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x2;
              break;
            case 3:
              compute_function =
                  (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
              channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x3;
              break;
            case 4:
              compute_function =
                  (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
              channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x4;
              break;
            default:
              compute_function =
                  (pthreadpool_function_1d_t)compute_channel_shuffle_variable;
              channel_shuffle_context.variable_ukernel =
                  pytorch_qnnp_params.x8zip.xm;
              break;
            case 0:
            case 1:
              PYTORCH_QNNP_UNREACHABLE;
          }
          pthreadpool_compute_1d(
              threadpool,
              compute_function,
              &channel_shuffle_context,
              op->batch_size);
          break;
        }
        default:
          PYTORCH_QNNP_UNREACHABLE;
      }
      return pytorch_qnnp_status_success;
        */
}
