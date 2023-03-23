crate::ix!();

pub struct Clip {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Clip);
    base: NeuralNetOperator,

    min: f32,
    max: f32,
}

impl Clip {

    pub fn new(min: f32, max: f32) -> Self {
    
        todo!();
        /*
            : NeuralNetOperator(NNKind::Clip), min_(min), max_(max)
        */
    }
    
    #[inline] pub fn get_min(&self) -> f32 {
        
        todo!();
        /*
            return min_;
        */
    }
    
    #[inline] pub fn get_max(&self) -> f32 {
        
        todo!();
        /*
            return max_;
        */
    }
    
    #[inline] pub fn set_min(&mut self, min: f32)  {
        
        todo!();
        /*
            min_ = min;
        */
    }
    
    #[inline] pub fn set_max(&mut self, max: f32)  {
        
        todo!();
        /*
            max_ = max;
        */
    }
}

