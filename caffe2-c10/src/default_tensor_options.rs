crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DefaultTensorOptions.h]

/**
  | Like TensorOptions, but all fields
  | are guaranteed to be filled.
  |
  */
#[derive(Default)]
pub struct DefaultTensorOptions {

    /**
      | = TypeMeta::Make<float>(); // 64-bit
      |
      */
    dtype:         TypeMeta,

    /**
      | 32-bit
      |
      */
    device:        Device, // default = kCPU

    /**
      | 8-bit
      |
      */
    layout:        Layout, // default = kStrided

    /**
      | 8-bit
      |
      */
    requires_grad: bool, // default = false
}

impl DefaultTensorOptions {

    pub fn dtype(&self) -> TypeMeta {
        
        todo!();
        /*
            return dtype_;
        */
    }
    
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return device_;
        */
    }
    
    pub fn layout(&self) -> std::alloc::Layout {
        
        todo!();
        /*
            return layout_;
        */
    }
    
    pub fn requires_grad(&self) -> bool {
        
        todo!();
        /*
            return requires_grad_;
        */
    }

    /// Defined in TensorOptions.h
    ///
    #[inline] pub fn merge(&mut self, options: &TensorOptions) -> &mut DefaultTensorOptions {
        
        todo!();
        /*
        
        */
    }
}

#[inline] pub fn get_default_tensor_options<'a>() -> &'a DefaultTensorOptions {
    
    todo!();
        /*
            static const auto options = DefaultTensorOptions();
      return options;
        */
}
