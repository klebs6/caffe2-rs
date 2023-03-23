crate::ix!();

pub struct Concat {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Concat);
    base: NeuralNetOperator,

    axis:     i32,
    add_axis: bool,
}

impl Concat {
    
    pub fn new(
        axis: Option<i32>, 
        add_axis: Option<bool>) -> Self {

        let axis: i32 = axis.unwrap_or(-1);
        let add_axis: bool = add_axis.unwrap_or(false);

        todo!();
        /*
            : NeuralNetOperator(NNKind::Concat), axis_(axis), addAxis_(addAxis)
        */
    }
    
    #[inline] pub fn get_axis(&self) -> i32 {
        
        todo!();
        /*
            return axis_;
        */
    }
    
    #[inline] pub fn get_add_axis(&self) -> bool {
        
        todo!();
        /*
            return addAxis_;
        */
    }
    
    #[inline] pub fn set_axis(&mut self, axis: i32)  {
        
        todo!();
        /*
            axis_ = axis;
        */
    }
    
    #[inline] pub fn set_add_axis(&mut self, add_axis: bool)  {
        
        todo!();
        /*
            addAxis_ = addAxis;
        */
    }
}

