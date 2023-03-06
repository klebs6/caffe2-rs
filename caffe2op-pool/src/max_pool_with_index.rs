crate::ix!();

#[USE_CONV_POOL_BASE_FUNCTIONS("CUDAContext")]
pub struct MaxPoolWithIndexOp {

    base: ConvPoolOpBase<CUDAContext>,

    /*
      | Input: X
      | 
      | Output: Y, mask
      |
      */
}

impl MaxPoolWithIndexOp {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CUDAContext>(operator_def, ws)
        */
    }
}

#[USE_CONV_POOL_BASE_FUNCTIONS("CUDAContext")]
pub struct MaxPoolWithIndexGradientOp {

    base: ConvPoolOpBase<CUDAContext>,

    /*
      | Input: X, dY, mask
      | 
      | Output: dX
      |
      */
}

impl MaxPoolWithIndexGradientOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<CUDAContext>(operator_def, ws)
        */
    }
}
