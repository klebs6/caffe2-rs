crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/operator.h]

pub enum PyTorchQnnpFormat {
    pytorch_qnnp_format_quint8 = 0x02000000,
    pytorch_qnnp_format_float32 = 0x02020202,
    pytorch_qnnp_format_float16 = 0x01010101,
}

pub enum PyTorchQnnpUKernelType {
    pytorch_qnnp_ukernel_type_none = 0,
    pytorch_qnnp_ukernel_type_add,
    pytorch_qnnp_ukernel_type_average_pooling,
    pytorch_qnnp_ukernel_type_channel_shuffle,
    pytorch_qnnp_ukernel_type_clamp,
    pytorch_qnnp_ukernel_type_conv,
    pytorch_qnnp_ukernel_type_dwconv,
    pytorch_qnnp_ukernel_type_gemm,
    pytorch_qnnp_ukernel_type_gemm_sparse_dq,
    pytorch_qnnp_ukernel_type_gemm_prepackA_sparse_dq,
    pytorch_qnnp_ukernel_type_global_average_pooling,
    pytorch_qnnp_ukernel_type_lut,
    pytorch_qnnp_ukernel_type_max_pooling,
    pytorch_qnnp_ukernel_type_softargmax,
    pytorch_qnnp_ukernel_type_xzp_gemm,
}

pub struct SparseMatrix {
    col_indices:    *const u32,
    row_values:     *const u32,
    values:         *const u8,
    row_block_size: u32,
    col_block_size: u32,
}

pub union PyTorchQnnpOperatorUnion {
    requantization_params:       PyTorchQnnpQ31RequantizationParams,
    conv_quantization_params:    PyTorchQnnpConvQuantizationParams,
    add_quantization_params:     PyTorchQnnpAddQuantizationParams,
    avgpool_quantization_params: PyTorchQnnpAvgPoolQuantizationParams,
    u8_clamping_params:          PyTorchQnnpU8ClampingParams,
}

pub struct PyTorchQnnpOperator {
    batch_size:                       Size,
    input_padding_top:                u32,
    input_padding_right:              u32,
    input_padding_bottom:             u32,
    input_padding_left:               u32,
    adjustment_height:                u32,
    adjustment_width:                 u32,
    kernel_height:                    u32,
    kernel_width:                     u32,
    stride_height:                    u32,
    stride_width:                     u32,
    dilation_height:                  u32,
    dilation_width:                   u32,
    groups:                           u32,
    group_stride:                     Size,
    group_channels:                   Size,
    group_input_channels:             Size,
    group_output_channels:            Size,
    channels:                         Size,
    input_height:                     Size,
    input_width:                      Size,
    input_pixel_stride:               Size,
    input:                            *const void,
    indirection_buffer:               *const *const void,
    a_sum:                            *mut void,
    input2_pixel_stride:              Size,
    input2:                           *const void,
    output_height:                    Size,
    output_width:                     Size,
    output_pixel_stride:              Size,
    output:                           *mut void,
    packed_weights:                   *mut void,
    input_scale:                      f32,
    output_scale:                     f32,
    input_zero_point:                 u8,
    kernel_zero_point:                u8,
    output_zero_point:                u8,
    output_min:                       u8,
    output_max:                       u8,
    valid_batch_size:                 Size,
    last_input_height:                Size,
    last_input_width:                 Size,
    last_input:                       *const void,
    zero_buffer:                      *mut void,
    zero_pointer:                     *mut void,
    lookup_table:                     *mut void,
    u:                                PyTorchQnnpOperatorUnion,

    ukernel_type:                     PyTorchQnnpUKernelType,
    format:                           PyTorchQnnpFormat,
    per_channel:                      bool,

    /**
      | Sparsity support
      |
      */
    sparse_matrix:                    SparseMatrix,
    bias:                             *const void,
    dynamic_conv_quantization_params: PyTorchQnnpConvDynamicQuantizationParams,
    prepacked_a:                      *mut u8,
}

#[inline] pub fn pytorch_qnnp_operator_get_log2_output_element_size(convolution: *const PytorchQnnpOperator) -> u32 {
    
    todo!();
        /*
            return (u32)(convolution->format & UINT32_C(0xFF));
        */
}

#[inline] pub fn pytorch_qnnp_operator_get_log2_input_element_size(convolution: *const PytorchQnnpOperator) -> u32 {
    
    todo!();
        /*
            return (u32)((convolution->format >> 8) & UINT32_C(0xFF));
        */
}

#[inline] pub fn pytorch_qnnp_operator_get_log2_kernel_element_size(convolution: *const PytorchQnnpOperator) -> u32 {
    
    todo!();
        /*
            return (u32)((convolution->format >> 16) & UINT32_C(0xFF));
        */
}

#[inline] pub fn pytorch_qnnp_operator_get_log2_bias_element_size(convolution: *const PytorchQnnpOperator) -> u32 {
    
    todo!();
        /*
            return (u32)((convolution->format >> 24) & UINT32_C(0xFF));
        */
}
