crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DefaultDtype.h]
//-------------------------------------------[.cpp/pytorch/c10/core/DefaultDtype.cpp]

lazy_static!{
    /*
    static auto default_dtype = TypeMeta::Make<float>();
    */
}

lazy_static!{
    /*
    static auto default_dtype_as_scalartype = default_dtype.toScalarType();
    */
}

lazy_static!{
    /*
    static auto default_complex_dtype =
        TypeMeta::Make<complex<float>>();
    */
}

pub fn set_default_dtype(dtype: TypeMeta)  {
    
    todo!();
        /*
            default_dtype = dtype;
      default_dtype_as_scalartype = default_dtype.toScalarType();
      switch (default_dtype_as_scalartype) {
        case ScalarType::f16:
          default_complex_dtype = ScalarType::Complexf16;
          break;
        case ScalarType::Double:
          default_complex_dtype = ScalarType::ComplexDouble;
          break;
        default:
          default_complex_dtype = ScalarType::ComplexFloat;
          break;
      }
        */
}

pub fn get_default_dtype() -> TypeMeta {
    
    todo!();
        /*
            return default_dtype;
        */
}

pub fn get_default_dtype_as_scalartype() -> ScalarType {
    
    todo!();
        /*
            return default_dtype_as_scalartype;
        */
}

pub fn get_default_complex_dtype() -> TypeMeta {
    
    todo!();
        /*
            return default_complex_dtype;
        */
}
