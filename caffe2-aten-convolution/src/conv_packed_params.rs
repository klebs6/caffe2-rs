crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/conv_packed_params.h]

pub trait ConvPackedParamsBaseInterface<const SPATIAL_DIM: i32 = 2>:
CustomClassHolder
+ Apply
+ ApplyRelu
+ Unpack
+ Stride
+ Padding
+ OutputPadding
+ Dilation
+ Groups
+ Transpose {}

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

pub trait Unpack {

    fn unpack(&mut self) -> (Tensor,Option<Tensor>);
}

pub trait Stride {

    fn stride(&self) -> TorchList<i64>;
}

pub trait Padding {
    
    fn padding(&self) -> TorchList<i64>;
}

pub trait OutputPadding {
    
    fn output_padding(&self) -> TorchList<i64>;
}

pub trait Dilation {
    
    fn dilation(&self) -> TorchList<i64>;
}

pub trait Groups {
    
    fn groups(&self) -> i64;
}

pub trait Transpose {
    
    fn transpose(&self) -> bool;
}
