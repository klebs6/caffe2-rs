/*!
  | In order to preserve bc, we make
  | 
  | DeprecatedTypeProperties instances
  | unique just like they are for Type.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/DeprecatedTypePropertiesRegistry.h]

pub struct DeprecatedTypePropertiesDeleter {

}

impl DeprecatedTypePropertiesDeleter {
    
    pub fn invoke(&mut self, ptr: *mut DeprecatedTypeProperties)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct DeprecatedTypePropertiesRegistry {
    registry: [[Box<DeprecatedTypeProperties>; Backend::NumOptions]; ScalarType::NumOptions],
}

impl DeprecatedTypePropertiesRegistry {
    
    pub fn get_deprecated_type_properties(&self, 
        p: Backend,
        s: ScalarType) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
        
        */
    }
    
    pub fn global_deprecated_type_properties_registry(&mut self) -> &mut DeprecatedTypePropertiesRegistry {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/DeprecatedTypePropertiesRegistry.cpp]

impl DeprecatedTypePropertiesDeleter {
    
    pub fn invoke(&mut self, ptr: *mut DeprecatedTypeProperties)  {
        
        todo!();
        /*
            delete ptr;
        */
    }
}

impl DeprecatedTypePropertiesRegistry {
    
    pub fn new() -> Self {
    
        todo!();
        /*


            for (int b = 0; b < static_cast<int>(Backend::NumOptions); ++b) {
        for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); ++s) {
          registry[b][s] = make_unique<DeprecatedTypeProperties>(
                  static_cast<Backend>(b),
                  static_cast<ScalarType>(s));
        }
      }
        */
    }
    
    pub fn get_deprecated_type_properties(&self, 
        p: Backend,
        s: ScalarType) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return *registry[static_cast<int>(p)][static_cast<int>(s)];
        */
    }
}

/**
  | TODO: This could be bad juju if someone calls
  | globalContext() in the destructor of an object
  | with static lifetime.
  |
  */
pub fn global_deprecated_type_properties_registry() -> &mut DeprecatedTypePropertiesRegistry {
    
    todo!();
        /*
            static DeprecatedTypePropertiesRegistry singleton;
      return singleton;
        */
}
