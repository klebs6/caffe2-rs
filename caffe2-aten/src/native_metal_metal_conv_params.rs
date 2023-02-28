crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalConvParams.h]

#[derive(Default)]
pub struct Conv2DParams {

    /**
      | batch size
      |
      */
    N:  i64,


    /**
      | channels
      |
      */
    C:  i64,


    /**
      | input height
      |
      */
    H:  i64,


    /**
      | input width
      |
      */
    W:  i64,


    /**
      | output channels
      |
      */
    OC: i64,


    /**
      | input channels
      |
      */
    IC: i64,


    /**
      | kernel height
      |
      */
    KH: i64,


    /**
      | kernel width
      |
      */
    KW: i64,


    /**
      | stride y (height)
      |
      */
    SY: i64,


    /**
      | stride x (width)
      |
      */
    SX: i64,


    /**
      | padding y (height)
      |
      */
    PY: i64,


    /**
      | padding x (width)
      |
      */
    PX: i64,


    /**
      | dilation y (height)
      |
      */
    DY: i64,


    /**
      | dilation x (width)
      |
      */
    DX: i64,


    /**
      | groups
      |
      */
    G:  i64,


    /**
      | output width
      |
      */
    OW: i64,


    /**
      | output height
      |
      */
    OH: i64,
}

impl Conv2DParams {
    
    pub fn new(
        input_sizes:  &[i32],
        weight_sizes: &[i32],
        padding:      &[i32],
        stride:       &[i32],
        dilation:     &[i32],
        groups:       i64) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn output_sizes(&self) -> Vec<i64> {
        
        todo!();
        /*
            return {N, OC, OH, OW};
        */
    }
    
    pub fn is_depthwise(&self) -> bool {
        
        todo!();
        /*
            // Currently, only channel multipler of 1 is supported
        // i.e. inputFeatureChannels == outputFeatureChannels
        return G > 1 && IC == 1 && OC == G && OC == C;
        */
    }
}

