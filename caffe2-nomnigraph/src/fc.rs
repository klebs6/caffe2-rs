crate::ix!();

pub struct FC {
    //NOMNIGRAPH_DEFINE_NN_RTTI(FC);
    base: NeuralNetOperator,

    axis:  i32,
    axisW: i32,
}

impl FC {

    pub fn new(
        axis: Option<i32>, 
        axisW: Option<i32>) -> Self 
    {
        let axis: i32 = axis.unwrap_or(1);
        let axisW: i32 = axisW.unwrap_or(1);

        todo!();
        /*
            : NeuralNetOperator(NNKind::FC), axis_(axis), axisW_(axisW)
        */
    }
    
    #[inline] pub fn get_axis(&self) -> i32 {
        
        todo!();
        /*
            return axis_;
        */
    }
    
    #[inline] pub fn get_axisW(&self) -> i32 {
        
        todo!();
        /*
            return axisW_;
        */
    }
    
    #[inline] pub fn set_axis(&mut self, axis: i32)  {
        
        todo!();
        /*
            axis_ = axis;
        */
    }
    
    #[inline] pub fn set_axisW(&mut self, axisW: i32)  {
        
        todo!();
        /*
            axisW_ = axisW;
        */
    }
}

