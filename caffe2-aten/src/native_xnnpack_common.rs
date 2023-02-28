crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/Common.h]
pub struct Deleter {

}

impl Deleter {
    
    pub fn invoke(&self, op: XnnOperator)  {
        
        todo!();
        /*
            xnn_delete_operator(op);
        */
    }
}

pub type Operator = Box<XnnOperator,Deleter>;

pub struct ContextLinear {
    op:              Operator,
    output_channels: i64,
}

pub mod context_linear {

    pub const K_MIN: f32 = -f32::infinity;
    pub const K_MAX: f32 = f32::infinity;
}

impl ContextLinear {
    
    pub fn new(
        o:          Operator,
        o_channels: i64) -> Self {
    
        todo!();
        /*
        op = move(o);
        output_channels = o_channels;
        */
    }
}

/**
  | This contains information for both
  | the transpose and non-transpose cases.
  |
  */
pub struct ContextConv2D {
    op:                Operator,
    weight_size:       [i64; 4],
    padding:           [i64; 2],
    output_padding:    [i64; 2],
    stride:            [i64; 2],
    dilation:          [i64; 2],
    cached_input_ptr:  *const f32, // default = { nullptr }
    cached_output_ptr: *const f32, // default = { nullptr }
    input_height:      usize, // default = { 0 }
    input_width:       usize, // default = { 0 }
    batch_size:        usize, // default = { 0 }
    input_channels:    usize, // default = { 0 }
    transposed:        bool,
    groups:            i64,
}

pub mod context_conv2d {

    pub const K_MIN: f32 = -f32::infinity;
    pub const K_MAX: f32 = f32::infinity;
}

impl ContextConv2D {
    
    pub fn new(
        o:              Operator,
        weight_size:    [i64; 4],
        padding:        [i64; 2],
        output_padding: [i64; 2],
        stride:         [i64; 2],
        dilation:       [i64; 2],
        transposed:     bool,
        groups:         i64) -> Self {
    
        todo!();
        /*


            :  op(move(o)),
             weight_size_(weight_size),
             padding_(padding),
             output_padding_(output_padding),
             stride_(stride),
             dilation_(dilation),
             transposed_(transposed),
             groups_(groups)
        */
    }
}

pub mod layout {

    use super::*;

    /// 4D Activation Maps
    ///
    pub mod activation4d {
        pub const BATCH:    usize = 0;
        pub const CHANNELS: usize = 1;
        pub const HEIGHT:   usize = 2;
        pub const WIDTH:    usize = 3;
    }

    /// ND Activation Maps
    pub struct ActivationND {

    }

    impl ActivationND {

        /**
          | Some operators may not be limited to
          | 4 dimensional tensors.
          |
          | In that scenario, XNNPACK denotes that
          | operator with an _nc suffix and expects
          | all dimensions, except channels, to be
          | flattened into one argument:
          | batch_size.
          |
          */
        pub fn batch(tensor: &[i32]) -> i64 {
            
            todo!();
            /*
                if (C10_UNLIKELY(tensor.empty())) {
                        return -1;
                    }

                    // Handle the case where batch size is zero.
                    i64 batch = tensor[0];

                    for (usize index = 1u; index < (tensor.size() - 1u); ++index) {
                        batch *= tensor[index];
                    }

                    return batch;
                }{
            */
        }
        
        pub fn channel(tensor: &[i32]) -> i64 {
            
            todo!();
            /*
                if (C10_UNLIKELY(tensor.empty())) {
                        return -1;
                    }

                    return tensor.back();
                }{
            */
        }
    }

    /// Convolution Filters
    ///
    pub mod filter {

        pub const OUTPUT: usize = 0;
        pub const INPUT:  usize = 1;
        pub const HEIGHT: usize = 2;
        pub const WIDTH:  usize = 3;
    }

    /// Parameters (Pooling Kernels, Dilation,
    /// Padding, Stride, etc.)
    ///
    pub mod parameter {
        pub const HEIGHT: usize = 0;
        pub const WIDTH:  usize = 1;
    }
}

pub fn available() -> bool {
    
    todo!();
        /*
        
        */
}
