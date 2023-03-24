crate::ix!();

#[macro_export] macro_rules! register_external_tensor_functions {
    ($id:expr, $($arg:expr),*) => {
        /*
        
          C10_REGISTER_TYPED_CLASS(ExternalTensorFunctionsBaseRegistry, id, __VA_ARGS__)
        */
    }
}

#[inline] pub fn create_external_tensor_functions(id: TypeIdentifier) -> Box<dyn ExternalTensorFunctionsBase> {
    
    todo!();
    /*
        return ExternalTensorFunctionsBaseRegistry()->Create(id);
    */
}
