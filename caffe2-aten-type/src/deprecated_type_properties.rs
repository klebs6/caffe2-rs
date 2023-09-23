crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/DeprecatedTypeProperties.h]

/**
  | This class specifies a Backend and
  | a ScalarType. Currently, it primarily serves as
  | a replacement return value for Tensor::type().
  |
  | Previously, Tensor::type() returned Type&, but
  | we are changing Type to not be dtype-specific.
  */
pub struct DeprecatedTypeProperties {
    backend:     Backend,
    scalar_type: ScalarType,
}

impl PartialEq<DeprecatedTypeProperties> for DeprecatedTypeProperties {
    
    #[inline] fn eq(&self, other: &DeprecatedTypeProperties) -> bool {
        todo!();
        /*
            return backend_ == other.backend() && scalar_type_ == other.scalarType();
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/DeprecatedTypeProperties.cpp]
impl DeprecatedTypeProperties {
    
    pub fn new(
        backend:     Backend,
        scalar_type: ScalarType) -> Self {
    
        todo!();
        /*
        : backend(backend),
        : scalar_type(scalar_type),

        
        */
    }
    
    pub fn backend(&self) -> Backend {
        
        todo!();
        /*
            return backend_;
        */
    }
    
    pub fn layout(&self) -> Layout {
        
        todo!();
        /*
            return layout_from_backend(backend_);
        */
    }
    
    pub fn is_sparse(&self) -> bool {
        
        todo!();
        /*
            return layout_from_backend(backend()) == kSparse;
        */
    }
    
    pub fn is_sparse_csr(&self) -> bool {
        
        todo!();
        /*
            return layout_from_backend(backend()) == kSparseCsr;
        */
    }
    
    pub fn device_type(&self) -> DeviceType {
        
        todo!();
        /*
            return backendToDeviceType(backend_);
        */
    }
    
    pub fn is_cuda(&self) -> bool {
        
        todo!();
        /*
            return backendToDeviceType(backend_) == kCUDA;
        */
    }
    
    pub fn scalar_type(&self) -> ScalarType {
        
        todo!();
        /*
            return scalar_type_;
        */
    }
    
    pub fn type_meta(&self) -> TypeMeta {
        
        todo!();
        /*
            return scalarTypeToTypeMeta(scalar_type_);
        */
    }
    
    pub fn to_string(&self) -> String {
        
        todo!();
        /*
            string base_str;
        if (backend_ == Backend::Undefined || scalar_type_ == ScalarType::Undefined) {
          base_str = "UndefinedType";
        } else {
          base_str = string(toString(backend_)) + toString(scalar_type_) + "Type";
        }
        return base_str;
        */
    }
    
    pub fn to_backend(&self, b: Backend) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
            b, scalar_type_);
        */
    }
    
    pub fn to_scalar_type(&self, s: ScalarType) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
            backend_, s);
        */
    }
    
    pub fn cpu(&self) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return toBackend(Backend::CPU);
        */
    }
    
    pub fn cuda(&self) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return toBackend(Backend::CUDA);
        */
    }
    
    pub fn hip(&self) -> &mut DeprecatedTypeProperties {
        
        todo!();
        /*
            return toBackend(Backend::HIP);
        */
    }

    /**
      | Constructs the `TensorOptions` from
      | a type and a `device_index`.
      |
      */
    pub fn options_with_device_index(&self, device_index: Option<i16>) -> TensorOptions {

        let device_index: i16 = device_index.unwrap_or(-1);

        todo!();
        /*
            return TensorOptions().dtype(typeMeta())
                              .device(device_type(), device_index)
                              .layout(layout());
        */
    }

    /**
      | Constructs the `TensorOptions` from a type
      | and a Device.  Asserts that the device type
      | matches the device type of the type.
      |
      */
    pub fn options(&self, device_opt: Option<Device>) -> TensorOptions {
        
        todo!();
        /*
            if (!device_opt.has_value()) {
          return options(-1);
        } else {
          Device device = device_opt.value();
          AT_ASSERT(device.type() == device_type());
          return options(device.index());
        }
        */
    }
    
    pub fn operator_tensor_options(&self) -> TensorOptions {
        
        todo!();
        /*
            return options();
        */
    }
    
    pub fn id(&self) -> i64 {
        
        todo!();
        /*
            return static_cast<i64>(backend()) *
            static_cast<i64>(ScalarType::NumOptions) +
            static_cast<i64>(scalarType());
        */
    }
    
    pub fn unsafe_tensor_fromth(&self, 
        th_pointer: *mut c_void,
        retain:     bool) -> Tensor {
        
        todo!();
        /*
            return unsafeTensorFromTH(th_pointer, retain);
        */
    }
    
    pub fn unsafe_storage_fromth(&self, 
        th_pointer: *mut c_void,
        retain:     bool) -> Storage {
        
        todo!();
        /*
            return unsafeStorageFromTH(th_pointer, retain);
        */
    }
    
    pub fn copy_(&self, 
        src:          &Tensor,
        non_blocking: Option<bool>,
        to_device:    Option<Device>) -> Tensor {

        let non_blocking: bool = non_blocking.unwrap_or(false);
        
        todo!();
        /*
            if (to_device) {
        return src.to(src.options().dtype(scalarType()).device(to_device), non_blocking, /*copy=*/true);
      }
      return src.to(src.options().dtype(scalarType()), non_blocking, /*copy=*/true);
        */
    }
}
