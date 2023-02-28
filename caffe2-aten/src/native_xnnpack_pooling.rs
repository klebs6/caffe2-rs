crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/Pooling.h]

pub struct PoolingParameters {
    kernel:   [i64; 2],
    padding:  [i64; 2],
    stride:   [i64; 2],
    dilation: [i64; 2],
}

impl PoolingParameters {
    
    pub fn new(
        kernel:   &[i32],
        padding:  &[i32],
        stride:   &[i32],
        dilation: &[i32]) -> Self {
    
        todo!();
        /*
        : kernel(normalize(kernel_)),
        : padding(normalize(padding_)),
        : stride(normalize(stride_)),
        : dilation(normalize(dilation_)),

        
        */
    }
    
    pub fn normalize(parameter: &[i32]) -> [i64; 2] {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(
                !parameter.empty(),
                "Invalid usage!  Reason: normalize() was called on an empty parameter.");

            return array<i64, 2>{
                parameter[0],
                (2 == parameter.size()) ? parameter[1] : parameter[0],
            };
        */
    }
}
