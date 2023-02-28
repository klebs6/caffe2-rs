crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/packed_params.h]

pub trait LinearPackedParamsBaseInterface:
TorchJitCustomClassHolder
+ Apply
+ ApplyRelu
+ ApplyDynamic
+ ApplyDynamicRelu
+ Unpack
+ Bias {

    /// out variant of LinearPackedParamsBase::apply
    fn apply_out(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &mut Tensor {
        
        todo!();
        /*
            throw runtime_error(
            "apply_out is not implemented for this packed "
            "parameter type");
        return output;
        */
    }
    
    fn apply_relu_out(&mut self, 
        input:             &Tensor,
        output_scale:      f64,
        output_zero_point: i64,
        output:            &mut Tensor) -> &mut Tensor {
        
        todo!();
        /*
            throw runtime_error(
            "apply_relu_out is not implemented for this packed "
            "parameter type");
        return output;
        */
    }
    
    fn set_bias(&mut self, bias: Option<Tensor>)  {
        
        todo!();
        /*
            throw runtime_error(
            "set_bias is not implemented for this packed "
            "parameter type");
        */
    }
}

pub trait Apply {

    
    fn apply(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor;
}

pub trait ApplyRelu {

    
    fn apply_relu(&mut self, 
        input:             Tensor,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor;
}

pub trait ApplyDynamic {

    
    fn apply_dynamic(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor;
}

pub trait ApplyDynamicRelu {

    
    fn apply_dynamic_relu(&mut self, 
        input:        Tensor,
        reduce_range: bool) -> Tensor;
}

pub trait Unpack {

    
    fn unpack(&mut self) -> (Tensor,Option<Tensor>);
}

pub trait Bias {

    
    fn bias(&mut self) -> Option<Tensor>;
}
