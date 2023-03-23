crate::ix!();

#[macro_export] macro_rules! nomnigraph_define_nn_rtti {
    ($op:ident) => {
        todo!();
        /*
        
          static bool classof(const NeuralNetOperator* N) {        
            return N->getKind() == NNKind::op;                                
          }                                                                   
          static bool classof(const Value* N) {                    
            if (isa<NeuralNetOperator>(N)) {                                  
              return dyn_cast<NeuralNetOperator>(N)->getKind() == NNKind::op; 
            }                                                                 
            return false;                                                     
          }
        */
    }
}
