crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/QuantizerBase.h]

pub type QuantizerPtr = IntrusivePtr<Quantizer>;

/**
  | Each concrete Quantizer type should
  | have a unique QScheme type.
  |
  */
pub trait QuantizerInterface:
Qscheme
+ Quantize
+ Dequantize
+ EqualTo {}

pub trait Qscheme {

    fn qscheme(&self) -> QScheme;
}

pub trait Quantize {

    /**
      | quantize a float Tensor into a quantized
      | Tensor.
      |
      */
    fn quantize(&mut self, t: &Tensor) -> Tensor;
}

pub trait Dequantize {

    /**
      | dequantize a quantized Tensor into
      | a float Tensor.
      |
      */
    fn dequantize(&mut self, t: &Tensor) -> Tensor;
}

pub trait EqualTo {

    /**
      | Compare against `other` for equality.
      |
      */
    fn equal_to(&mut self, other: QuantizerPtr) -> bool;
}

/**
 | Quantizer is the class for storing all the
 | information that's necessary to perform
 | quantize and dequantize operation.
 |
 | We might have different types of quantization
 | schemes and this is the base class for all
 | quantizers.
 |
 | QTensorImpl will hold a pointer to Quantizer so
 | that we can support different quantization
 | schemes on Tensor.
 |
 | For example, the most common quantization
 | scheme, Affine Quantization, requires scale and
 | zero_point as parameters, we'll store scale and
 | zero_point inside the instance and we can use
 | it to quantize a float Tensor or dequantize
 | a quantized Tensor.
 |
 | When you add new types of leaf Quantizer class,
 | please also make sure to add a corresponding
 | QScheme enum since they should have one to one
 | mapping.
 |
 | Note about intrusive_ptr:
 |
 | Quantized Tensor holds an intrusive_ptr to
 | Quantizer, and multiple Tensor can share the
 | same Quantizer. Quantizer should be immutable.
 */
pub struct Quantizer {
    base:        IntrusivePtrTarget,
    scalar_type: ScalarType,
}

impl Quantizer {

    pub fn new(scalar_type: ScalarType) -> Self {
    
        todo!();
        /*
        : scalar_type(scalar_type),

        
        */
    }

    /// Copied from torch/csrc/jit/ir/scope.h
    pub fn intrusive_from_this(&mut self) -> QuantizerPtr {
        
        todo!();
        /*
            raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                               // from a raw `this` pointer
                                               // so we need to bump the refcount
                                               // to account for this ownership
        return intrusive_ptr<Quantizer>::reclaim(this);
        */
    }
    
    pub fn scalar_type(&mut self) -> ScalarType {
        
        todo!();
        /*
            return scalar_type_;
        */
    }
}
