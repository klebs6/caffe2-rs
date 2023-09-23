/*!
  | This file provides implementations of
  | InlineDeviceGuard and
  | InlineOptionalDeviceGuard.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/impl/InlineDeviceGuard.h]

/**
 | A DeviceGuard is an RAII class that sets
 | a device to some value on construction, and
 | resets the device to its original value on
 | destruction.
 |
 | InlineDeviceGuard is a helper class for
 | implementing DeviceGuards.
 |
 | It is templated over a DeviceGuardImpl
 | (anything that implements
 | DeviceGuardImplInterface).  There are two
 | primary ways to instantiate InlineDeviceGuard:
 |
 |  - With a concrete implementation of
 |  DeviceGuardImpl, e.g., CUDAGuardImpl.
 |
 |    This is the best way to use
 |    InlineDeviceGuard, as all calls are
 |    devirtualized, giving you code as efficient
 |    as straight line calls to
 |    cudaGetDevice/cudaSetDevice.
 |
 |  - With VirtualGuardImpl, which does a virtual
 |    dispatch to a DeviceGuardImpl retrieved from
 |    a DeviceType registry.  We have explicitly
 |    instantiated InlineDeviceGuard this way as
 |    DeviceGuard.
 |
 | If you are in a hurry, you can use
 | InlineDeviceGuard directly:
 |
 |    using CUDAGuard = InlineDeviceGuard<CUDAGuardImpl>;
 |
 | However, you can provide a better user
 | experience if you explicitly write a wrapper
 | class that itself contains the template
 | instantiation:
 |
 |    class CUDAGuard {
 |    
 |      // ... the API ...
 |    
 |      InlineDeviceGuard<CUDAGuardImpl> guard_;
 |    }
 |
 | The wrapper class provides a good place to
 | write documentation, and helps avoid weird
 | template instantiation errors when a user
 | incorrectly uses the class.
 |
 | If you need to test this class, consider
 | instantiating it with FakeGuardImpl.
 |
 | Note [Omitted default constructor from RAII]
 | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | In principle, we could add a default
 | constructor to DeviceGuard which reads the
 | current device and promises to restore to
 | that device on exit.  However, most cases
 | where you would have written this, you
 | probably meant to actually just use
 | OptionalDeviceGuard (since you don't actually
 | need the restore to happen if you don't ever
 | actually set the device).
 |
 | We remove the constructor here to encourage
 | you to think about what you actually want to
 | happen.
 |
 */
pub struct InlineDeviceGuard<T> {
    impl_:           T,
    original_device: Device,
    current_device:  Device,
}

impl<T> Drop for InlineDeviceGuard<T> {

    fn drop(&mut self) {
        todo!();
        /*
            impl_.uncheckedSetDevice(original_device_);
        */
    }
}

impl<T> InlineDeviceGuard<T> {

    /**
      | Set the current device to the passed
      | Device.
      |
      */
    pub fn new_from_device(device: Device) -> Self {
    
        todo!();
        /*


            : impl_(device.type()),
            original_device_(
                device.index() == -1 ? impl_.getDevice()
                                     : impl_.exchangeDevice(device)),
            current_device_(device.index() == -1 ? original_device_ : device)
        */
    }

    /**
      | Set the current device index to the passed
      | DeviceIndex.  (The device type is inferred
      | from the template parameter T).
      |
      | template<typename U = T, typename = typename enable_if<!is_same<U, VirtualGuardImpl>::value>::type>
      */
    pub fn new_from_index(device_index: DeviceIndex) -> Self {
    
        todo!();
        /*


            : InlineDeviceGuard(Device(U::static_type, device_index))
        */
    }

    /**
      | Construct an InlineDeviceGuard using
      | VirtualGuardImpl with an explicit
      | DeviceGuardImplInterface pointer.
      |
      */
    //template<typename U = T, typename = typename enable_if< is_same<U, VirtualGuardImpl>::value>::type>
    pub fn new_with_guard(
        device: Device,
        impl_:  *const dyn DeviceGuardImplInterface) -> Self {
    
        todo!();
        /*


            : impl_(
                VirtualGuardImpl(impl ? impl : getDeviceGuardImpl(device.type()))),
            original_device_(
                device.index() == -1 ? impl_.getDevice()
                                     : impl_.exchangeDevice(device)),
            current_device_(device.index() == -1 ? original_device_ : device)
        */
    }

    /**
      | Sets the device to the given one.
      | 
      | template<typename U = T, typename enable_if<!is_same<U,
      | VirtualGuardImpl>::value, int>::
      | type = 0>
      |
      */
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            AT_ASSERT(
            (U::static_type == DeviceType::HIP && device.is_cuda()) ||
            device.type() == U::static_type);
        auto index = device.index();
        if (index == -1)
          return;
        impl_.setDevice(device);
        current_device_ = device;
        */
    }

    /**
      | Resets the currently set device to its
      | original device, and then sets the current
      | device to the passed device.
      |
      | This is effectively equivalent to set_device
      | when a guard supports only a single device
      | type.
      |
      */
    //template <typename U = T> typename enable_if<!is_same<U, VirtualGuardImpl>::value>::type
    pub fn new_from_device_with_not_vguard(device: Device) -> Self {
    
        todo!();
        /*
            set_device(device);
        */
    }

    /**
      | Resets the currently set device to its
      | original device, and then sets the current
      | device to the passed device (for a possibly
      | different device type).
      |
      | This method is named reset_device to
      | highlight the fact that previous device
      | settings from this guard are NOT preserved,
      | even if the device has a different device
      | type.  For example:
      |
      |   // Cuda device is 0
      |   DeviceGuard g(Device(kCUDA, 1));
      |   g.reset_device(Device(kHIP, 2));
      |   // Cuda device is 0 (!!)
      |
      | NOTE: this implementation may skip some
      | device setting if it can prove that it is
      | unnecessary.
      |
      | Optional argument is for testing only.
      |
      */
    // template<typename U = T> typename enable_if<is_same<U, VirtualGuardImpl>::value>::type
    pub fn new_from_device_with_maybe_guard(
        device: Device,
        impl_:  Option<*const dyn DeviceGuardImplInterface>) -> Self {

        todo!();

        /*
        auto index = device.index();

        if (index == -1)
          return;
        if (device.type() == original_device_.type()) {
          AT_ASSERT(impl == nullptr || impl->type() == device.type());
          impl_.setDevice(device);
          current_device_ = device;
        } else {
          // Destruct and reconstruct the DeviceGuard in place
          impl_.setDevice(original_device_);
          impl_ = !impl ? VirtualGuardImpl(device.type()) : VirtualGuardImpl(impl);
          original_device_ = impl_.exchangeDevice(device);
          current_device_ = device;
        }
        */
    }

    /**
      | Sets the device index to the given one.  The
      | device type is inferred from the original
      | device type.
      |
      */
    pub fn set_index(&mut self, index: DeviceIndex)  {
        
        todo!();
        /*
            reset_device(Device(original_device_.type(), index));
        */
    }

    /**
      | Returns the device that was set at the time
      | the most recent reset_device(), or otherwise
      | the device at construction time.
      |
      */
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return original_device_;
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
      | construction, or via
      | set_device/reset_device/set_index.
      |
      */
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return current_device_;
        */
    }
}

/**
  | A OptionalDeviceGuard is an RAII class
  | that sets a device to some value on initialization,
  | and resets the device to its original
  | value on destruction.
  | 
  | InlineOptionalDeviceGuard is a helper
  | class for implementing
  | 
  | OptionalDeviceGuards. See guidance
  | in InlineDeviceGuard on how to use this.
  | See OptionalDeviceGuard for user-oriented
  | usage notes.
  |
  | Note [Explicit initialization of optional fields]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Explicit initialization of optional fields
  | required to workaround an nvcc bug; see
  | https://github.com/pytorch/pytorch/issues/12117
  */
pub struct InlineOptionalDeviceGuard<T> {
    guard: Option<InlineDeviceGuard<T>>,
}

impl<T> InlineOptionalDeviceGuard<T> {

    /// Creates an uninitialized
    /// OptionalDeviceGuard.
    ///
    pub fn new_default() -> Self {
    
        todo!();
        /*


            : guard_() // See Note [Explicit initialization of optional fields]
        */
    }

    /// Set the current device to the passed
    /// Device, if it is not nullopt.
    ///
    pub fn new_from_maybe_deviceopt(device_opt: Option<Device>) -> Self {
    
        todo!();
        /*


            : guard_() 
              // See Note [Explicit initialization of optional fields]
        if (device_opt.has_value()) {
          guard_.emplace(device_opt.value());
        }
        */
    }

    /// Set the current device to the passed
    /// DeviceIndex, if it is not nullopt.
    ///
    /// template<typename U = T, typename = typename enable_if< !is_same<U, VirtualGuardImpl>::value>::type>
    ///
    pub fn new_from_maybe_deviceindex(device_index_opt: Option<DeviceIndex>) -> Self {
    
        todo!();
        /*


            : guard_() 
        // See Note [Explicit initialization of optional fields]
        if (device_index_opt.has_value()) {
          guard_.emplace(device_index_opt.value());
        }
        */
    }

    /**
      | All constructors of DeviceGuard are valid
      | for OptionalDeviceGuard and result in
      | initialized OptionalDeviceGuard.
      |
      */
    pub fn new_from_args<Args>(args: Args) -> Self {
    
        todo!();
        /*


            : guard_(in_place, forward<Args>(args)...)
        */
    }

    /*
      | TODO: Consider readding Tensor and &[Tensor]
      | constructors here, when Tensor moves to c10.
      | (These are only valid on OptionalDeviceGuard,
      | because a Tensor may be undefined, in which
      | case we need an uninitialized tensor guard.)
      |
      | Note [Move construction for RAII guards is tricky]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      | In principle, move construction is useful for
      | terminating the lifetime of
      | a `OptionalDeviceGuard` early; for example:
      |
      |     // current device is d0
      |     OptionalDeviceGuard g1(d1);
      |     // current device is d1
      |     {
      |       OptionalDeviceGuard g2(move(g1));
      |     }
      |     // current device is d0!!
      |
      | However, it's difficult to implement the move
      | constructor in a way that works in all
      | situations.  For example, consider the
      | following example:
      |
      |     OptionalDeviceGuard g1(d1);
      |     {
      |       OptionalDeviceGuard g2(d2);
      |       {
      |         OptionalDeviceGuard g3(move(g1)); // !!!
      |       }
      |     }
      |
      | What should the current device be while g3 in
      | scope... and what should it be after it goes
      | out of scope?  What about g2? There don't
      | seem to be satisfactory answers for these
      | questions.
      |
      | It's in principle possible to raise an error
      | when this occurs by doing some extra
      | thread-local bookkeeping.  But why bother?
      | Just don't provide the constructor.
      |
      | Note [Move assignment for RAII guards is tricky]
      | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      | Move assignment is deleted, because you need
      | to know which guard was defined "first", as
      | that guard's original_device_ wins--with the
      | current representation, we have no way of
      | telling which is the case.  (Move
      | construction does not have this problem, as
      | one guard is always uninitialized.)
      |
      | We can make this clear by way of a pair of
      | examples:
      |
      | Example 1:
      |
      |  // initial device is n0
      |  {
      |    CUDAGuard g1(n1);
      |    {
      |      CUDAGuard g2(n2);
      |      // current device should be n2
      |      g1 = move(g2);
      |      // current device should still be n2
      |    }
      |    // current device should still be n2
      |  }
      |  // current device should be n0
      |
      |  Example 2 (flip the order of the two guards):
      |
      |  // initial device is n0
      |  {
      |    CUDAGuard g2(n2);
      |    {
      |      CUDAGuard g1(n1);
      |      // current device should be n1
      |      g1 = move(g2);
      |      // current device should be n2
      |    }
      |    // current device should be n0 (since g2 has been vacated)
      |  }
      |
      | In both examples, we need g1 to restore to n0
      | after move assignment.
      |
      | However, in example 1, this is determined by
      | the restore value of g1 (prior to the
      | move). In example 2, however, it is
      | determined by the the restore value of
      | g2(!!). We don't know which one should win,
      | without having a way of telling which guard
      | was allocated first.
      |
      | We could solve this with an extra
      | thread-local variable.  But no one is
      | actually using move-assignment.  So just get
      | rid of it.
      |
      */

    /**
      | Sets the device to the given one.
      |
      | Initializes OptionalDeviceGuard if it is not
      | already initialized.
      |
      | template<typename U = T, typename = typename enable_if<!is_same<U, VirtualGuardImpl>::value>::type>
      */
    pub fn set_device(&mut self, device: Device)  {
        
        todo!();
        /*
            if (!guard_.has_value()) {
          guard_.emplace(device);
        } else {
          guard_->set_device(device);
        }
        */
    }

    /**
      | Resets the currently set device to its
      | original device, and then sets the current
      | device to the passed device (for a possibly
      | different device type).
      |
      | Initializes OptionalDeviceGuard if it is not
      | already initialized.
      |
      | See notes on why this is called reset_device
      | on InlineDeviceGuard.
      |
      | Optional argument is for testing only.
      */
    // template < typename U = T, typename = typename enable_if< is_same<U, VirtualGuardImpl>::value>::type>
    pub fn reset_device_with_guard(&mut self, 
        device: Device,
        impl_:  Option<*const dyn DeviceGuardImplInterface>)  {

        todo!();
        /*
            if (!guard_.has_value()) {
          guard_.emplace(device, impl);
        } else {
          guard_->reset_device(device, impl);
        }
        */
    }

    /**
      | Resets the currently set device to its
      | original device, and then sets the current
      | device to the passed device.  Initializes
      | the guard if it is not already initialized.
      |
      | This is effectively equivalent to set_device
      | when a guard supports only a single device
      | type.
      |
      */
  //template < typename U = T, typename = typename enable_if< !is_same<U, VirtualGuardImpl>::value>::type>
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            if (!guard_.has_value()) {
          guard_.emplace(device);
        } else {
          guard_->reset_device(device);
        }
        */
    }

    /**
      | Sets the device index to the given one.
      | The device type is statically known.
      |
      */
    /// template < typename U = T, typename = typename enable_if< !is_same<U, VirtualGuardImpl>::value>::type>
    pub fn set_index(&mut self, index: DeviceIndex)  {
        
        todo!();
        /*
            if (!guard_.has_value()) {
          guard_.emplace(index);
        } else {
          guard_->set_index(index);
        }
        */
    }

    /**
      | Returns the device that was set immediately
      | prior to initialization of the, guard, or
      | nullopt if the guard is uninitialized.
      |
      */
    pub fn original_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.has_value() ? make_optional(guard_->original_device())
                                  : nullopt;
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
      | construction, or via set_device, if the
      | guard is initialized, or nullopt if the
      | guard is uninitialized.
      |
      */
    pub fn current_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.has_value() ? make_optional(guard_->current_device())
                                  : nullopt;
        */
    }

    /**
      | Restore the original device, resetting
      | this guard to uninitialized state.
      |
      */
    pub fn reset(&mut self)  {
        
        todo!();
        /*
            guard_.reset();
        */
    }
}
