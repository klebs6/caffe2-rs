crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/MetaTensor.cpp]

/**
  | The meta allocator ignores whatever
  | allocation is requested and always
  | gives you nullptr
  |
  */
#[derive(Default)]
pub struct MetaAllocator {
    base: Allocator,
}

impl MetaAllocator {
    
    pub fn deleter(pointer: *mut void)  {
        
        todo!();
        /*
            TORCH_INTERNAL_ASSERT(!pointer);
        */
    }
    
    pub fn allocate(&self, nbytes: Size) -> DataPtr {
        
        todo!();
        /*
            return {nullptr, nullptr, &deleter, Device(DeviceType_Meta)};
        */
    }
    
    pub fn raw_deleter(&self) -> DeleterFnPtr {
        
        todo!();
        /*
            return deleter;
        */
    }
}

lazy_static!{
    /*
    static MetaAllocator g_meta_alloc;
    */
}

pub fn get_meta_allocator() -> *mut Allocator {
    
    todo!();
        /*
            return &g_meta_alloc;
        */
}

pub fn empty_meta(
        size:              &[i32],
        dtype_opt:         Option<ScalarType>,
        layout_opt:        Option<Layout>,
        device_opt:        Option<Device>,
        pin_memory_opt:    Option<bool>,
        memory_format_opt: Option<MemoryFormat>) -> Tensor {
    
    todo!();
        /*
            auto device = device_or_default(device_opt);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType_Meta);
      // NB: because there is no SparseMeta (yet), non-strided layout is
      // exerciseable
      TORCH_CHECK_NOT_IMPLEMENTED(
        layout_or_default(layout_opt) == Layout::Strided,
        "strided meta tensors not supported yet"
      );

      auto* allocator = GetMetaAllocator();
      auto dtype = dtype_or_default(dtype_opt);
      auto r = empty_generic(size, allocator, DispatchKey::Meta, dtype, device, memory_format_opt);
      return r;
        */
}

pub fn empty_strided_meta(
        size:           &[i32],
        stride:         &[i32],
        dtype_opt:      Option<ScalarType>,
        layout_opt:     Option<Layout>,
        device_opt:     Option<Device>,
        pin_memory_opt: Option<bool>) -> Tensor {
    
    todo!();
        /*
            auto t = native::empty_meta({0}, dtype_opt, layout_opt, device_opt, pin_memory_opt);
      // Amazingly the CPU implementation will work for us, because most of resize
      // is generic except the memcpy, but the memcpy will be skipped if the source
      // storage is nullptr (which it always is, for meta tensors)
      native::resize_impl_cpu_(t.unsafeGetTensorImpl(), size, stride);
      return t;
        */
}
