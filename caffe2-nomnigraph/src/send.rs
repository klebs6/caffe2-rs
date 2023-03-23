crate::ix!();

pub struct Send_ {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Send_);
    base: NeuralNetOperator,
    destination: String,
}

impl Send_ {

    pub fn new(destination: String) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Send_), destination_(destination)
        */
    }
    
    #[inline] pub fn get_destination(&self) -> String {
        
        todo!();
        /*
            return destination_;
        */
    }
    
    #[inline] pub fn set_destination(&mut self, destination: String)  {
        
        todo!();
        /*
            destination_ = std::move(destination);
        */
    }
}

