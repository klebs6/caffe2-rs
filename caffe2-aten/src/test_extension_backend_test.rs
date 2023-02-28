crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/extension_backend_test.cpp]

lazy_static!{
    /*
    static int test_int;
    */
}

pub fn empty_override(
    size:                   &[i32],
    dtype:                  Option<ScalarType>,
    layout:                 Option<Layout>,
    device:                 Option<Device>,
    pin_memory:             Option<bool>,
    optional_memory_format: Option<MemoryFormat>) -> Tensor {

    todo!();
        /*
            test_int = 1;
      auto tensor_impl = make_intrusive<TensorImpl, UndefinedTensorImpl>(
          Storage(
              Storage::use_byte_size_t(),
              0,
              DataPtr(nullptr, Device(DeviceType_MSNPU, 1)),
              nullptr,
              false),
          DispatchKey::MSNPU,
          TypeMeta::Make<float>());
      return Tensor(move(tensor_impl));
        */
}

pub fn add_override(
        a: &Tensor,
        b: &Tensor,
        c: &Scalar) -> Tensor {
    
    todo!();
        /*
            test_int = 2;
      return a;
        */
}

pub fn empty_strided_override(
        size:       &[i32],
        stride:     &[i32],
        dtype:      Option<ScalarType>,
        layout:     Option<Layout>,
        device:     Option<Device>,
        pin_memory: Option<bool>) -> Tensor {
    
    todo!();
        /*
            return empty_override(size, dtype, layout, device, pin_memory, nullopt);
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, MSNPU, m) {
      m.impl("empty.memory_format",  empty_override);
      m.impl("empty_strided",        empty_strided_override);
      m.impl("add.Tensor",           add_override);
    }
    */
}

#[test] fn backend_extension_test_register_op() {
    todo!();
    /*
    
      Tensor a = empty({5, 5}, kMSNPU);
      ASSERT_EQ(a.device().type(), kMSNPU);
      ASSERT_EQ(a.device().index(), 1);
      ASSERT_EQ(a.dtype(), TypeMeta::Make<float>());
      ASSERT_EQ(test_int, 1);

      Tensor b = empty_like(a, kMSNPU);
      ASSERT_EQ(b.device().type(), kMSNPU);
      ASSERT_EQ(b.device().index(), 1);
      ASSERT_EQ(b.dtype(), TypeMeta::Make<float>());

      add(a, b);
      ASSERT_EQ(test_int, 2);

      // Ensure that non-MSNPU operator still works
      Tensor d = empty({5, 5}, kCPU);
      ASSERT_EQ(d.device().type(), kCPU);

    */
}
