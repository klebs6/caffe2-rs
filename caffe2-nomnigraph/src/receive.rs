crate::ix!();

pub struct Receive {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Receive);
    base: NeuralNetOperator,
    source: String,
}

impl Receive {

    pub fn new(source: String) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::Receive), source_(source)
        */
    }
    
    #[inline] pub fn get_source(&self) -> String {
        
        todo!();
        /*
            return source_;
        */
    }
    
    #[inline] pub fn set_source(&mut self, source: String)  {
        
        todo!();
        /*
            source_ = std::move(source);
        */
    }
}

