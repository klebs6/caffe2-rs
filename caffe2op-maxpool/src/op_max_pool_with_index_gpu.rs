crate::ix!();

use crate::{
    ConvPoolOpBase,
    CUDAContext,
    OperatorDef,
    Workspace
};

pub struct MaxPoolWithIndexOp {

    //USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
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

pub struct MaxPoolWithIndexGradientOp {

    //USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
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
