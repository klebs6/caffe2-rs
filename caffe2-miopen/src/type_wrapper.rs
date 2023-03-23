crate::ix!();

pub trait miopenTypeWrapper {
    type BNParamType;
    type ScalingParamType;
}

/**
  | miopenTypeWrapper is a wrapper class
  | that allows us to refer to the miopen
  | type in a template function.
  | 
  | The class is specialized explicitly
  | for different data types below.
  |
  */
pub struct miopenTypeWrapperF32 { }

impl miopenTypeWrapper for miopenTypeWrapperF32 {
    type BNParamType      = f32;
    type ScalingParamType = f32;
}

impl miopenTypeWrapperF32 {

    //const type_: miopenDataType_t = miopenDataType_t::miopenFloat;
    
    #[inline] pub fn k_one() -> *mut ScalingParamType {
        
        todo!();
        /*
            static ScalingParamType v = 1.0;
        return &v;
        */
    }
    
    #[inline] pub fn k_zero() -> *const ScalingParamType {
        
        todo!();
        /*
            static ScalingParamType v = 0.0;
        return &v;
        */
    }
}

///------------------------------
pub struct miopenTypeWrapperHalf { }

impl miopenTypeWrapper for miopenTypeWrapperHalf {
    type BNParamType      = f32;
    type ScalingParamType = f32;
}

impl miopenTypeWrapperHalf {

    //const type_: miopenDataType_t = miopenHalf;
    
    #[inline] pub fn k_one() -> *mut ScalingParamType {
        
        todo!();
        /*
            static ScalingParamType v = 1.0;
            return &v;
        */
    }
    #[inline] pub fn k_zero() -> *mut ScalingParamType {
        
        todo!();
        /*
            static ScalingParamType v = 0.0;
            return &v;
        */
    }

}

