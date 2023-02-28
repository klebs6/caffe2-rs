crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/core/impl/InlineStreamGuard_test.cpp]

pub const TestDeviceType: DeviceType = DeviceType::Cuda;

pub type TestGuardImpl = FakeGuardImpl<TestDeviceType>;

pub fn dev(index: DeviceIndex) -> Device {
    
    todo!();
        /*
            return Device(TestDeviceType, index);
        */
}

pub fn stream(
        index: DeviceIndex,
        sid:   StreamId) -> Stream {
    
    todo!();
        /*
            return Stream(Stream::UNSAFE, dev(index), sid);
        */
}

// -- InlineStreamGuard -------------------------------------------------------

pub type TestGuard = InlineStreamGuard<TestGuardImpl>;

#[test] fn inline_stream_guard_constructor() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        TestGuard g(stream(1, 2));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), stream(0, 0));
        ASSERT_EQ(g.current_stream(), stream(1, 2));
        ASSERT_EQ(g.original_device(), dev(0));
        ASSERT_EQ(g.current_device(), dev(1));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

#[test] fn inline_stream_guard_reset_same_device() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        TestGuard g(stream(0, 2));
        g.reset_stream(stream(0, 3));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 3);
        ASSERT_EQ(g.original_stream(), stream(0, 0));
        ASSERT_EQ(g.current_stream(), stream(0, 3));
        ASSERT_EQ(g.original_device(), dev(0));
        ASSERT_EQ(g.current_device(), dev(0));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

#[test] fn inline_stream_guard_reset_different_same_device() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        TestGuard g(stream(1, 2));
        g.reset_stream(stream(1, 3));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), stream(0, 0));
        ASSERT_EQ(g.current_stream(), stream(1, 3));
        ASSERT_EQ(g.original_device(), dev(0));
        ASSERT_EQ(g.current_device(), dev(1));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

#[test] fn inline_stream_guard_reset_different_device() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        TestGuard g(stream(1, 2));
        g.reset_stream(stream(2, 3));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), stream(0, 0));
        ASSERT_EQ(g.current_stream(), stream(2, 3));
        ASSERT_EQ(g.original_device(), dev(0));
        ASSERT_EQ(g.current_device(), dev(2));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

// -- OptionalInlineStreamGuard
// -------------------------------------------------------

pub type OptionalTestGuard = InlineOptionalStreamGuard<TestGuardImpl>;

#[test] fn inline_optional_stream_guard_constructor() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        OptionalTestGuard g(stream(1, 2));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
        ASSERT_EQ(g.current_stream(), make_optional(stream(1, 2)));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      {
        OptionalTestGuard g(make_optional(stream(1, 2)));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
        ASSERT_EQ(g.current_stream(), make_optional(stream(1, 2)));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      {
        OptionalTestGuard g;
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

#[test] fn inline_optional_stream_guard_reset_same_device() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        OptionalTestGuard g;
        g.reset_stream(stream(1, 3));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
        ASSERT_EQ(g.current_stream(), make_optional(stream(1, 3)));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

#[test] fn inline_optional_stream_guard_reset_different_device() {
    todo!();
    /*
    
      TestGuardImpl::setDeviceIndex(0);
      TestGuardImpl::resetStreams();
      {
        OptionalTestGuard g;
        g.reset_stream(stream(2, 3));
        ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(g.original_stream(), make_optional(stream(0, 0)));
        ASSERT_EQ(g.current_stream(), make_optional(stream(2, 3)));
      }
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);

    */
}

// -- InlineMultiStreamGuard
// -------------------------------------------------------

pub type MultiTestGuard = InlineMultiStreamGuard<TestGuardImpl>;

#[test] fn inline_multi_stream_guard_constructor() {
    todo!();
    /*
    
      TestGuardImpl::resetStreams();
      {
        vector<Stream> streams;
        MultiTestGuard g(streams);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      }
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      {
        vector<Stream> streams = {stream(0, 2)};
        MultiTestGuard g(streams);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      }
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      {
        vector<Stream> streams = {stream(1, 3)};
        MultiTestGuard g(streams);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
      }
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
      {
        vector<Stream> streams = {stream(0, 2), stream(1, 3)};
        MultiTestGuard g(streams);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
        ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
      }
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
      ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);

    */
}
