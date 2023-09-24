crate::ix!();


pub struct LinearPackedParamsBase {
    base:                    CustomClassHolder,
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

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ao_sparse/quantized/cpu/packed_params.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/packed_params.h]

pub trait LinearPackedParamsBaseInterface:
CustomClassHolder
+ Apply
+ ApplyDynamic
+ ApplyDynamicRelu
+ ApplyDynamicReluWithReduceRange
+ ApplyDynamicWithReduceRange
+ ApplyOut
+ ApplyRelu
+ ApplyReluOut
+ Bias
+ SetBias 
+ Unpack
{ }

pub trait ApplyOut {

    /// out variant of LinearPackedParamsBase::apply
    fn apply_out<'a>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            throw runtime_error(
            "apply_out is not implemented for this packed "
            "parameter type");
        return output;
        */
    }
}

pub trait ApplyReluOut {

    fn apply_relu_out<'a>(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
        /*
            throw runtime_error(
            "apply_relu_out is not implemented for this packed "
            "parameter type");
        return output;
        */
    }
}


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

pub trait ApplyDynamicWithReduceRange {
    
    fn apply_dynamic_with_reduce_range(
        &mut self, 
        input:        Tensor,
        reduce_range: bool

    ) -> Tensor;
}

pub trait ApplyDynamicRelu {
    
    fn apply_dynamic_relu(&mut self, input: &Tensor) -> Tensor;
}

pub trait ApplyDynamicReluWithReduceRange {
    
    fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor;
}

/// <Weight, bias, out_features_block_size, in_features_block_size>
pub type LinearPackedSerializationType = (Tensor,Option<Tensor>,Vec<i64>);

pub trait Unpack {

    //type Output = (Tensor,Option<Tensor>)
    type Output = LinearPackedSerializationType;
    
    fn unpack(&mut self) -> ;
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
