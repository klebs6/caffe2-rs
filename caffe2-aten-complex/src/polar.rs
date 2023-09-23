crate::ix!();

pub fn polar_out<'a>(
        abs:    &Tensor,
        angle:  &Tensor,
        result: &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            complex_check_dtype(result, abs, angle);
      auto iter = TensorIteratorConfig()
          .add_output(result)
          .add_input(abs)
          .add_input(angle)
          .check_all_same_dtype(false)
          .build();
      polar_stub(iter.device_type(), iter);
      return result;
        */
}

pub fn polar(
        abs:   &Tensor,
        angle: &Tensor) -> Tensor {
    
    todo!();
        /*
            complex_check_floating(abs, angle);
      TensorOptions options = abs.options();
      options = options.dtype(toComplexType(abs.scalar_type()));
      Tensor result = empty(0, options);
      return polar_out(result, abs, angle);
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorFactories.h]
pub type BinaryFn = fn(_0: &mut TensorIterator) -> ();

declare_dispatch!{binary_fn, complex_stub}
declare_dispatch!{binary_fn, polar_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/TensorFactories.cpp]

define_dispatch!{complex_stub}
define_dispatch!{polar_stub}

/**
  | Temporary type cast operators. These are needed
  | to trace type-casts now since Type's are not
  | supported in the IR. Instead, we call down to
  | these specialized operators for each datatype.
  |
  | TODO: remove when we have Type support in the
  | IR
  */
#[macro_export] macro_rules! define_cast_op {
    ($_1:ident, $n:ident) => {
        /*
        
          Tensor _cast_##n(const Tensor& self, bool non_blocking) {      
            if (self.scalar_type() == ScalarType::n)                     
              return self;                                               
            return self.to(ScalarType::n, non_blocking);                 
          }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types_and3!{Bool, Half, BFloat16, DEFINE_CAST_OP}
    */
}
