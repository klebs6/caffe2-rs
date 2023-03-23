crate::ix!();

pub struct GenericOperator {
    //NOMNIGRAPH_DEFINE_NN_RTTI(GenericOperator);
    base: NeuralNetOperator,
    name: String,
}

impl Default for GenericOperator {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::GenericOperator
        */
    }
}

impl GenericOperator {
    
    pub fn new(name: String) -> Self {
    
        todo!();
        /*
            : NeuralNetOperator(NNKind::GenericOperator), name_(name)
        */
    }
    
    #[inline] pub fn set_name(&mut self, name: String)  {
        
        todo!();
        /*
            name_ = name;
        */
    }
}

impl Named for GenericOperator {

    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            return name_;
        */
    }
}
