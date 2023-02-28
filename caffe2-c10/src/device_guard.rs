/*!
  | Note [Whither the DeviceGuard boilerplate]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Design note: in principle, we could avoid these
  | wrappers using:
  |
  | using DeviceGuard = InlineDeviceGuard<VirtualGuardImpl>;
  | using OptionalDeviceGuard =
  | InlineOptionalDeviceGuard<VirtualGuardImpl>;
  |
  | But the error messages are worse, and our users
  | can't just look at the header file to find out
  | what's going on.  Furthermore, for
  | specializations like CudaStreamGuard, it can be
  | profitable to replace some interfaces with
  | refined types (e.g., return CudaStream instead
  | of Stream).  So, we eat the boilerplate and
  | write out the API explicitly.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DeviceGuard.h]

/**
  | RAII guard that sets a certain default device
  | in its constructor, and changes it back to the
  | device that was originally active upon
  | destruction.
  |
  | The device is always reset to the one that was
  | active at the time of construction of the
  | guard. Even if you `set_device` after
  | construction, the destructor will still reset
  | the device to the one that was active at
  | construction time.
  |
  | This device guard does NOT have an
  | uninitialized state; it is guaranteed to reset
  | a device on exit.  If you are in a situation
  | where you *might* want to setup a guard (i.e.,
  | are looking for the moral equivalent of
  | optional<DeviceGuard>), see
  | OptionalDeviceGuard.
  */
pub struct DeviceGuard {
    guard: InlineDeviceGuard<VirtualGuardImpl>,
}

impl From<Device> for DeviceGuard {

    /// Set the current device to the passed
    /// Device.
    ///
    fn from(device: Device) -> Self {
    
        todo!();
        /*
        : guard(device),

        
        */
    }
}

impl DeviceGuard {

    /// This constructor is for testing only.
    ///
    pub fn new(
        device: Device,
        impl_:  Box<dyn DeviceGuardImplInterface>) -> Self {
    
        todo!();
        /*
        : guard(device, impl),
        */
    }

    /**
      | Sets the device to the given one.  The
      | specified device must be consistent with the
      | device type originally specified during
      | guard construction.
      |
      | TODO: The consistency check here is
      | inconsistent with StreamGuard's behavior
      | with set_stream, where a stream on
      | a different device than the original one
      | isn't an error; we just reset the stream and
      | then switch devices.
      */
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }

    /// This method is for testing only.
    ///
    pub fn reset_device_with_guard(&mut self, 
        device: Device,
        impl_:  Box<dyn DeviceGuardImplInterface>)  {
        
        todo!();
        /*
            guard_.reset_device(device, impl);
        */
    }

    /**
      | Sets the device index to the given one.  The
      | device type is inferred from the original
      | device type the guard was constructed with.
      |
      */
    pub fn set_index(&mut self, index: DeviceIndex)  {
        
        todo!();
        /*
            guard_.set_index(index);
        */
    }

    /**
      | Returns the device that was set at the
      | time the guard was constructed.
      |
      */
    pub fn original_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
      | construction, or via set_device.
      |
      */
    pub fn current_device(&self) -> Device {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
}

/**
 | A OptionalDeviceGuard is an RAII class that
 | sets a device to some value on initialization,
 | and resets the device to its original value on
 | destruction.
 |
 | Morally, a OptionalDeviceGuard is equivalent to
 | optional<DeviceGuard>, but with extra
 | constructors and methods as appropriate.
 |
 | Besides its obvious use (optionally applying
 | a DeviceGuard), OptionalDeviceGuard is often
 | also used for the following idiom:
 |
 |    OptionalDeviceGuard g;
 |    for (const auto& t : tensors) {
 |      g.set_device(t.device());
 |      do_something_with(t);
 |    }
 |
 | This usage is marginally more efficient than
 | constructing a DeviceGuard every iteration of
 | the for loop, as it avoids an unnecessary
 | device reset.
 |
 | Unlike DeviceGuard, a OptionalDeviceGuard may
 | be uninitialized.  This occurs when you use the
 | nullary constructor, or pass a nullopt to the
 | constructor.
 |
 | Uninitialized OptionalDeviceGuards do
 | *nothing*; they do not know what the original
 | device was and they do not reset on
 | destruction.  This is why original_device() and
 | current_device() return optional<Device> rather
 | than Device (as they do in DeviceGuard), and
 | also is why we didn't just provide
 | OptionalDeviceGuard by default and hide
 | DeviceGuard from users.
 |
 | The semantics of an OptionalDeviceGuard are
 | exactly explained by thinking of it as an
 | optional<DeviceGuard>.  In particular, an
 | initialized OptionalDeviceGuard doesn't restore
 | device to its value at construction; it
 | restores device to its value *at
 | initialization*.  So if you have the program:
 |
 |  setDevice(1);
 |  OptionalDeviceGuard g;
 |  setDevice(2);
 |  g.reset_device(Device(DeviceType::CUDA, 3));  // initializes!
 |
 | On destruction, g will reset device to 2,
 | rather than 1.
 |
 | An uninitialized OptionalDeviceGuard is
 | distinct from a (initialized) DeviceGuard whose
 | original_device_ and current_device_ match,
 | since the DeviceGuard will still reset the
 | device to original_device_.
 */
pub struct OptionalDeviceGuard {
    guard: InlineOptionalDeviceGuard<VirtualGuardImpl>,
}

impl Default for OptionalDeviceGuard {

    /**
      | Create an uninitialized guard. Set
      | the guard later using reset_device.
      |
      */
    fn default() -> Self {
    
        todo!();
        /*
        : guard(),

        
        */
    }
}

impl From<Option<Device>> for OptionalDeviceGuard {

    /**
      | Initialize the guard if a Device is passed;
      | otherwise leave the guard uninitialized.
      |
      */
    fn from(device: Option<Device>) -> Self {
    
        todo!();
        /*
        : guard(device),
        */
    }
}

impl OptionalDeviceGuard {

    /**
      | Constructor for testing only.
      |
      */
    pub fn new(
        device: Device,
        impl_:  *const dyn DeviceGuardImplInterface) -> Self {
    
        todo!();
        /*
        : guard(device, impl),

        
        */
    }

    /**
      | Sets the device to the given one.
      |
      | The specified device must be consistent with
      | the device type originally specified during
      | guard construction.
      |
      */
    pub fn reset_device(&mut self, device: Device)  {
        
        todo!();
        /*
            guard_.reset_device(device);
        */
    }

    /// For testing only
    pub fn reset_device_with_guard(&mut self, 
        device: Device,
        impl_:  *const dyn DeviceGuardImplInterface)  {
        
        todo!();
        /*
            guard_.reset_device(device, impl);
        */
    }

    /**
      | Returns the device that was set at the
      | time the guard was constructed.
      |
      */
    pub fn original_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.original_device();
        */
    }

    /**
      | Returns the most recent device that was set
      | using this device guard, either from
      | construction, or via reset_device.
      |
      */
    pub fn current_device(&self) -> Option<Device> {
        
        todo!();
        /*
            return guard_.current_device();
        */
    }
}
