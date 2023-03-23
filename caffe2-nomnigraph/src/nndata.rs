crate::ix!();

pub struct NeuralNetData {
    base: Data,
    kind: NNDataKind,
}

impl Default for NeuralNetData {
    
    fn default() -> Self {
        todo!();
        /*
            : kind_(NNDataKind::Generic
        */
    }
}

pub trait NeuralNetDataTrait {
    fn clone(&mut self) -> *mut NeuralNetData;
}

impl NeuralNetData {

    pub fn new(kind: NNDataKind) -> Self {
    
        todo!();
        /*
            : kind_(kind)
        */
    }
    
    #[inline] pub fn get_kind(&self) -> NNDataKind {
        
        todo!();
        /*
            return kind_;
        */
    }
}

impl Named for NeuralNetData {
    
    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            switch (getKind()) {
        case NNDataKind::Tensor: {
          return dyn_cast<Tensor>(this)->getName();
        }
        default:
          return "";
      }
        */
    }
}
