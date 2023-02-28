crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/packed_params.h]

pub trait LinearPackedParamsBaseInterface:
Apply
+ ApplyRelu
+ ApplyDynamic
+ ApplyDynamicRelu
+ Unpack
+ Bias
+ SetBias {}

pub trait Apply {
    
    fn apply(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor;
}

pub trait ApplyRelu {
    
    fn apply_relu(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor;
}

pub trait ApplyDynamic {
    
    fn apply_dynamic(&mut self, input: &Tensor) -> Tensor;
}

pub trait ApplyDynamicRelu {
    
    fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor;
}

pub trait Unpack {
    
    fn unpack(&mut self) -> LinearPackedSerializationType;
}

pub trait Bias {
    
    fn bias(&mut self) -> Option<Tensor>;
}

pub trait SetBias  {
    
    fn set_bias(&mut self, bias: &Option<Tensor>)  {
        
        todo!();
        /*
            throw runtime_error(
            "set_bias is not implemented for this packed "
            "parameter type");
        */
    }
}

/// <Weight, bias, out_features_block_size, in_features_block_size>
pub type LinearPackedSerializationType = (Tensor,Option<Tensor>,Vec<i64>);

pub struct LinearPackedParamsBase {
    base:                    TorchJitCustomClassHolder,
    out_features_block_size: i64,
    in_features_block_size:  i64,
}

impl LinearPackedParamsBase {
    
    pub fn new(
        out_features_block_size: i64,
        in_features_block_size:  i64) -> Self {
    
        todo!();
        /*
        : out_features_block_size(out_features_block_size),
        : in_features_block_size(in_features_block_size),

        
        */
    }
}
