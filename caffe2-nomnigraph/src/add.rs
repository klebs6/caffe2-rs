crate::ix!();

pub struct Add {
    //NOMNIGRAPH_DEFINE_NN_RTTI(Add);
    base: NeuralNetOperator,
    broadcast: i32,
}

impl Add {
    
    pub fn new(broadcast: Option<i32>) -> Self {
    
        let broadcast: i32 = broadcast.unwrap_or(0);

        todo!();
        /*
            : NeuralNetOperator(NNKind::Add), broadcast_(broadcast)
        */
    }
    
    #[inline] pub fn get_broadcast(&self) -> i32 {
        
        todo!();
        /*
            return broadcast_;
        */
    }
    
    #[inline] pub fn set_broadcast(&mut self, broadcast: i32)  {
        
        todo!();
        /*
            broadcast_ = broadcast;
        */
    }
}

