crate::ix!();

pub struct NomTensor {
    base: NeuralNetData,
    name:  String,
    type_: DataType,
}

impl NomTensor {

    pub fn new(name: String) -> Self {
    
        todo!();
        /*
            : NeuralNetData(NNDataKind::Tensor),
            name_(name),
            type_(DataType::Generic)
        */
    }
    
    #[inline] pub fn classof(d: *const NeuralNetData) -> bool {
        
        todo!();
        /*
            return D->getKind() == NNDataKind::Tensor;
        */
    }
    
    #[inline] pub fn clone(&mut self) -> *mut NeuralNetData {
        
        todo!();
        /*
            return new NomTensor(name_);
        */
    }
    
    #[inline] pub fn set_type(&mut self, ty: DataType)  {
        
        todo!();
        /*
            type_ = type;
        */
    }
    
    #[inline] pub fn get_type(&self) -> DataType {
        
        todo!();
        /*
            return type_;
        */
    }
    
    #[inline] pub fn set_name(&mut self, name: &String)  {
        
        todo!();
        /*
            name_ = name;
        */
    }
}

impl Named for NomTensor {

    #[inline] fn get_name(&self) -> String {
        
        todo!();
        /*
            return name_;
        */
    }
}

