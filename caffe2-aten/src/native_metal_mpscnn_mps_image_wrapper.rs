crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/mpscnn/MPSImageWrapper.h]

//#[API_AVAILABLE(ios(10.0), macos(10.13))]
pub struct MPSImageWrapper {
    image_sizes:    Vec<i64>,
    image:          *mut MPSImage, // default = nil
    buffer:         id<MTLBuffer>, // default = nil
    command_buffer: *mut MetalCommandBuffer, // default = nil
    delegate:       id<PTMetalCommandBuffer>, // default = nil
}

impl MPSImageWrapper {
    
    pub fn new(sizes: &[i32]) -> Self {
    
        todo!();
        /*


        
        */
    }
    
    pub fn copy_data_from_host(&mut self, input_data: *const f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn copy_data_to_host(&mut self, host_data: *mut f32)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allocate_storage(&mut self, sizes: &[i32])  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn allocate_temporary_storage(&mut self, 
        sizes:          &[i32],
        command_buffer: *mut MetalCommandBuffer)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_command_buffer(&mut self, buffer: *mut MetalCommandBuffer)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn command_buffer(&self) -> *mut MetalCommandBuffer {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_image(&mut self, image: *mut MPSImage)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn image(&self) -> *mut MPSImage {
        
        todo!();
        /*
        
        */
    }
    
    pub fn buffer(&self) -> id<MTLBuffer> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn synchronize(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn prepare(&mut self)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn release(&mut self)  {
        
        todo!();
        /*
        
        */
    }
}
