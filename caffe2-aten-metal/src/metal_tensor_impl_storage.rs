crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalTensorImplStorage.h]

#[derive(Default)]
pub struct MetalTensorImplStorage {
    impl_: Arc<Impl>,
}

impl MetalTensorImplStorage {
    
    pub fn new(sizes: &Vec<i64>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn new(
        sizes:   &Vec<i64>,
        strides: &Vec<i64>) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn defined(&self) -> bool {
        
        todo!();
        /*
        
        */
    }
    
    pub fn sizes(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
    
    pub fn strides(&self) -> &[i32] {
        
        todo!();
        /*
        
        */
    }
    
    pub fn dim(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn numel(&self) -> i64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_data_from_host(&mut self, input_data: *const f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn copy_data_to_host(&mut self, host: *mut f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn texture(&self) -> *mut MPSImageWrapper {
        
        todo!();
        /*
        
        */
    }
    
    pub fn impl_(&mut self) -> Arc<Impl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn impl_(&self) -> Arc<Impl> {
        
        todo!();
        /*
        
        */
    }
}
