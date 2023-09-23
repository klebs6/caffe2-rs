// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include/qnnpack_func.h]

pub struct PrePackConvWeights {
    packed_weights:  *mut void, // default = nullptr
    output_channels: i64,
}

impl Drop for PrePackConvWeights {

    fn drop(&mut self) {
        todo!();
        /*
            if (packed_weights_ != nullptr) {
          free(packed_weights_);
        }
        */
    }
}

impl PrePackConvWeights {
    
    pub fn new(
        conv_param:         &ConvParam,
        kernel_zero_points: *const u8,
        kernel:             *const u8,
        bias:               *const i32) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn get_packed_weights(&self)  {
        
        todo!();
        /*
            return packed_weights_;
        */
    }
    
    pub fn get_output_channels(&self) -> i64 {
        
        todo!();
        /*
            return output_channels_;
        */
    }
}

pub struct PackBMatrix {
    packed_weights:  *mut void, // default = nullptr
    input_channels:  Size,
    output_channels: Size,
}

impl Drop for PackBMatrix {

    fn drop(&mut self) {
        todo!();
        /*
            if (packed_weights_ != nullptr) {
          free(packed_weights_);
        }
        */
    }
}

impl PackBMatrix {
    
    pub fn new(
        input_channels:       Size,
        output_channels:      Size,
        kernel_zero_points:   *const u8,
        requantization_scale: *const f32,
        kernel:               *const u8,
        bias:                 *const i32) -> Self {
    
        todo!();
        /*


        
        */
    }

    /**
      | This constructor is to be used for dynamic
      | mode quantization.
      |
      | In dynamic mode, we dont yet support per
      | channel quantization, and paying the cost of
      | memory allocation for per channel zero point
      | and requant scale will hurt performance.
      */
    pub fn new(
        input_channels:       Size,
        output_channels:      Size,
        kernel_zero_point:    u8,
        requantization_scale: f32,
        kernel:               *const u8,
        bias:                 *const i32) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn get_packed_weights(&self)  {
        
        todo!();
        /*
            return packed_weights_;
        */
    }
    
    pub fn get_input_channels(&self) -> Size {
        
        todo!();
        /*
            return input_channels_;
        */
    }
    
    pub fn get_output_channels(&self) -> Size {
        
        todo!();
        /*
            return output_channels_;
        */
    }
}

pub fn qnnpack_linear(
        batch_size:            Size,
        input_channels:        Size,
        output_channels:       Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        requantization_scales: *const f32,
        output_zero_point:     u8,
        output_min:            u8,
        output_max:            u8,
        input:                 *const u8,
        input_stride:          Size,
        packed_weights:        *mut void,
        output:                *mut u8,
        output_stride:         Size,
        threadpool:            threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}

pub fn qnnpack_conv(
        conv_p:                &ConvParam,
        convolution:           PyTorchQnnpOperator,
        packed_weights:        *mut void,
        batch_size:            Size,
        input_height:          Size,
        input_width:           Size,
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
        
        */
}

pub fn qnnpack_de_conv(
        deconv_p:              &ConvParam,
        deconvolution:         PyTorchQnnpOperator,
        packed_weights:        *mut void,
        batch_size:            Size,
        input_height:          Size,
        input_width:           Size,
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
        
        */
}

pub fn qnnpack_linear_dynamic(
        batch_size:            Size,
        input_channels:        Size,
        output_channels:       Size,
        input_zero_point:      u8,
        kernel_zero_points:    *const u8,
        dequantization_scales: *const f32,
        input:                 *const u8,
        input_stride:          Size,
        packed_weights:        *mut void,
        bias:                  *const f32,
        output:                *mut f32,
        output_stride:         Size,
        threadpool:            threadpool::ThreadPool) -> PyTorchQnnpStatus {
    
    todo!();
        /*
        
        */
}
