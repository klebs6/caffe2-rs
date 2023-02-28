crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/core/impl/InlineDeviceGuard_test.cpp]

pub const TestDeviceType: DeviceType = DeviceType::Cuda;

pub type TestGuardImpl = FakeGuardImpl<TestDeviceType>;

pub fn dev(index: DeviceIndex) -> Device {
    
    todo!();
        /*
            return Device(TestDeviceType, index);
        */
}

// -- InlineDeviceGuard -------------------------------------------------------

pub type TestGuard = InlineDeviceGuard<TestGuardImpl>;

#[test] fn inline_device_guard_constructor() {
    todo!();
    /*
    
      for (DeviceIndex i : {-1, 0, 1}) {
        DeviceIndex init_i = 0;
        TestGuardImpl::setDeviceIndex(init_i);
        auto test_body = [&](TestGuard& g) -> void {
          ASSERT_EQ(g.original_device(), dev(init_i));
          ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
          ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
          // Test un-bracketed write to device index
          TestGuardImpl::setDeviceIndex(4);
        };
        {
          // Index constructor
          TestGuard g(i);
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
        {
          // Device constructor
          TestGuard g(dev(i));
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
        /*
        {
          // Optional constructor
          TestGuard g(make_optional(dev(i)));
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
        */
      }

    */
}

#[test] fn inline_device_guard_constructor_error() {
    todo!();
    /*
    
      EXPECT_ANY_THROW(InlineDeviceGuard<FakeGuardImpl<DeviceType::CUDA>> g(
          Device(DeviceType::HIP, 1)));

    */
}

#[test] fn inline_device_guard_set() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      DeviceIndex i = init_i + 1;
      TestGuard g(i);
      DeviceIndex i2 = init_i + 2;
      g.set_device(dev(i2));
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
      g.set_device(dev(i2));
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);

    */
}

#[test] fn inline_device_guard_reset() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      DeviceIndex i = init_i + 1;
      TestGuard g(i);
      DeviceIndex i2 = init_i + 2;
      g.reset_device(dev(i2));
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
      g.reset_device(dev(i2));
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);

    */
}

#[test] fn inline_device_guard_set_index() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      DeviceIndex i = init_i + 1;
      TestGuard g(i);
      DeviceIndex i2 = init_i + 2;
      g.set_index(i2);
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
      g.set_index(i2);
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i2));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);

    */
}

// -- InlineOptionalDeviceGuard
// --------------------------------------------------

pub type MaybeTestGuard = InlineOptionalDeviceGuard<TestGuardImpl>;

#[test] fn inline_optional_device_guard_constructor() {
    todo!();
    /*
    
      for (DeviceIndex i : {-1, 0, 1}) {
        DeviceIndex init_i = 0;
        TestGuardImpl::setDeviceIndex(init_i);
        auto test_body = [&](MaybeTestGuard& g) -> void {
          ASSERT_EQ(g.original_device(), dev(init_i));
          ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
          ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
          // Test un-bracketed write to device index
          TestGuardImpl::setDeviceIndex(4);
        };
        {
          // Index constructor
          MaybeTestGuard g(i);
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
        {
          // Device constructor
          MaybeTestGuard g(dev(i));
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
        {
          // Optional constructor
          MaybeTestGuard g(make_optional(dev(i)));
          test_body(g);
        }
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
      }

    */
}

#[test] fn inline_optional_device_guard_nullary_constructor() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      auto test_body = [&](MaybeTestGuard& g) -> void {
        ASSERT_EQ(g.original_device(), nullopt);
        ASSERT_EQ(g.current_device(), nullopt);
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
      };
      {
        MaybeTestGuard g;
        test_body(g);
      }
      {
        // If you want nullopt directly to work, define a nullopt_t
        // overload.  But I don't really see why you'd want this lol.
        optional<Device> dev_opt = nullopt;
        MaybeTestGuard g(dev_opt);
        test_body(g);
      }

    */
}

#[test] fn inline_optional_device_guard_set() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      MaybeTestGuard g;
      DeviceIndex i = init_i + 1;
      g.set_device(dev(i));
      ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
      ASSERT_EQ(g.current_device(), make_optional(dev(i)));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
      g.set_device(dev(i));
      ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
      ASSERT_EQ(g.current_device(), make_optional(dev(i)));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);

    */
}

#[test] fn inline_optional_device_guard_set_index() {
    todo!();
    /*
    
      DeviceIndex init_i = 0;
      TestGuardImpl::setDeviceIndex(init_i);
      DeviceIndex i = init_i + 1;
      MaybeTestGuard g;
      g.set_index(i);
      ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
      ASSERT_EQ(g.current_device(), make_optional(dev(i)));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
      g.set_index(i);
      ASSERT_EQ(g.original_device(), make_optional(dev(init_i)));
      ASSERT_EQ(g.current_device(), make_optional(dev(i)));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);

    */
}

