// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include/pytorch_qnnpack.h]

/**
  | -----------
  | @brief
  | 
  | Status code for any QNNPACK function
  | call.
  |
  */
pub enum PyTorchQnnpStatus {

    /** The call succeeded, and all output arguments now contain valid data. */
    pytorch_qnnp_status_success               = 0,
    pytorch_qnnp_status_uninitialized         = 1,
    pytorch_qnnp_status_invalid_parameter     = 2,
    pytorch_qnnp_status_unsupported_parameter = 3,
    pytorch_qnnp_status_unsupported_hardware  = 4,
    pytorch_qnnp_status_out_of_memory         = 5,
}

pub fn pytorch_qnnp_initialize() -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_deinitialize() -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub type PyTorchQnnpOperator = *mut PyTorchQnnpOperator;

pub fn pytorch_qnnp_create_convolution2d_nhwc_q8(
        input_padding_top:     u32,
        input_padding_right:   u32,
        input_padding_bottom:  u32,
        input_padding_left:    u32,
        kernel_height:         u32,
        kernel_width:          u32,
        subsampling_height:    u32,
        subsampling_width:     u32,
        dilation_height:       u32,
        dilation_width:        u32,
        groups:                u32,
        group_input_channels:  Size,
        group_output_channels: Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        kernel:                *const u8,
        bias:                  *const i32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        flags:                 u32,
        requantization_scales: *const f32,
        per_channel:           bool,
        convolution:           *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_convolution2d_nhwc_q8(
        convolution:   PyTorchQnnpOperator,
        batch_size:    Size,
        input_height:  Size,
        input_width:   Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size,
        threadpool:    threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
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
        group_input_channels:  Size,
        group_output_channels: Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        kernel:                *const u8,
        bias:                  *const i32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        flags:                 u32,
        requantization_scales: *const f32,
        deconvolution:         *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
        deconvolution: PyTorchQnnpOperator,
        batch_size:    Size,
        input_height:  Size,
        input_width:   Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size,
        threadpool:    threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_fully_connected_nc_q8(
        input_channels:        Size,
        output_channels:       Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        kernel:                *const u8,
        bias:                  *const i32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        flags:                 u32,
        requantization_scales: *const f32,
        fully_connected:       *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
        input_channels:        Size,
        output_channels:       Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        kernel_col_indices:    *const u32,
        kernel_row_values:     *const u32,
        kernel_values:         *const u8,
        kernel_row_block_size: u32,
        kernel_col_block_size: u32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        flags:                 u32,
        requantization_scales: *const f32,
        use_prepack_kernel:    bool,
        fully_connected:       *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_fully_connected_nc_q8(
        fully_connected: PyTorchQnnpOperator,
        batch_size:      Size,
        input:           *const u8,
        input_stride:    Size,
        output:          *mut u8,
        output_stride:   Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
        fully_connected: PyTorchQnnpOperator,
        batch_size:      Size,
        input:           *const u8,
        input_stride:    Size,
        bias:            *const f32,
        output:          *mut f32,
        output_stride:   Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_global_average_pooling_nwc_q8(
        channels:               Size,
        input_zero_point:       u8,
        input_scale:            f32,
        output_zero_point:      u8,
        output_scale:           f32,
        output_min:             u8,
        output_max:             u8,
        flags:                  u32,
        global_average_pooling: *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_global_average_pooling_nwc_q8(
        global_average_pooling: PyTorchQnnpOperator,
        batch_size:             Size,
        width:                  Size,
        input:                  *const u8,
        input_stride:           Size,
        output:                 *mut u8,
        output_stride:          Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_average_pooling2d_nhwc_q8(
        input_padding_top:    u32,
        input_padding_right:  u32,
        input_padding_bottom: u32,
        input_padding_left:   u32,
        pooling_height:       u32,
        pooling_width:        u32,
        stride_height:        u32,
        stride_width:         u32,
        channels:             Size,
        input_zero_point:     u8,
        input_scale:          f32,
        output_zero_point:    u8,
        output_scale:         f32,
        output_min:           u8,
        output_max:           u8,
        flags:                u32,
        average_pooling:      *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
        average_pooling: PyTorchQnnpOperator,
        batch_size:      Size,
        input_height:    Size,
        input_width:     Size,
        input:           *const u8,
        input_stride:    Size,
        output:          *mut u8,
        output_stride:   Size,
        threadpool:      threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
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
        channels:             Size,
        output_min:           u8,
        output_max:           u8,
        flags:                u32,
        max_pooling:          *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
        max_pooling:   PyTorchQnnpOperator,
        batch_size:    Size,
        input_height:  Size,
        input_width:   Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size,
        threadpool:    threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_channel_shuffle_nc_x8(
        groups:          Size,
        group_channels:  Size,
        flags:           u32,
        channel_shuffle: *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_channel_shuffle_nc_x8(
        channel_shuffle: PyTorchQnnpOperator,
        batch_size:      Size,
        input:           *const u8,
        input_stride:    Size,
        output:          *mut u8,
        output_stride:   Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_add_nc_q8(
        channels:       Size,
        a_zero_point:   u8,
        a_scale:        f32,
        b_zero_point:   u8,
        b_scale:        f32,
        sum_zero_point: u8,
        sum_scale:      f32,
        sum_min:        u8,
        sum_max:        u8,
        flags:          u32,
        add:            *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_add_nc_q8(
        add:        PyTorchQnnpOperator,
        batch_size: Size,
        a:          *const u8,
        a_stride:   Size,
        b:          *const u8,
        b_stride:   Size,
        sum:        *mut u8,
        sum_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_clamp_nc_u8(
        channels:   Size,
        output_min: u8,
        output_max: u8,
        flags:      u32,
        clamp:      *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_clamp_nc_u8(
        clamp:         PyTorchQnnpOperator,
        batch_size:    Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_sigmoid_nc_q8(
        channels:          Size,
        input_zero_point:  u8,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        output_min:        u8,
        output_max:        u8,
        flags:             u32,
        sigmoid:           *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
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
        
        */
}

pub fn pytorch_qnnp_create_leaky_relu_nc_q8(
        channels:          Size,
        negative_slope:    f32,
        input_zero_point:  u8,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        output_min:        u8,
        output_max:        u8,
        flags:             u32,
        leaky_relu:        *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_leaky_relu_nc_q8(
        leaky_relu:    PyTorchQnnpOperator,
        batch_size:    Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_softargmax_nc_q8(
        channels:          Size,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        flags:             u32,
        softargmax:        *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
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
        
        */
}

pub fn pytorch_qnnp_create_tanh_nc_q8(
        channels:          Size,
        input_zero_point:  u8,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        output_min:        u8,
        output_max:        u8,
        flags:             u32,
        tanh:              *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_tanh_nc_q8(
        tanh:          PyTorchQnnpOperator,
        batch_size:    Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_hardsigmoid_nc_q8(
        channels:          Size,
        input_zero_point:  u8,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        output_min:        u8,
        output_max:        u8,
        flags:             u32,
        hardsigmoid:       *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_hardsigmoid_nc_q8(
        hardsigmoid:   PyTorchQnnpOperator,
        batch_size:    Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_create_hardswish_nc_q8(
        channels:          Size,
        input_zero_point:  u8,
        input_scale:       f32,
        output_zero_point: u8,
        output_scale:      f32,
        output_min:        u8,
        output_max:        u8,
        flags:             u32,
        hardswish:         *mut PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_setup_hardswish_nc_q8(
        hardswish:     PyTorchQnnpOperator,
        batch_size:    Size,
        input:         *const u8,
        input_stride:  Size,
        output:        *mut u8,
        output_stride: Size) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_run_operator(
        op:         PyTorchQnnpOperator,
        threadpool: threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn pytorch_qnnp_delete_operator(op: PyTorchQnnpOperator) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}
