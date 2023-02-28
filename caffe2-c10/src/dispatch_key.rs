crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/DispatchKey.h]

pub const DISPATCH_KEY_ZERO:      u8 = 0;
pub const DISPATCH_KEY_CATCHALL:  u8 = DISPATCH_KEY_ZERO;
pub const DISPATCH_KEY_UNDEFINED: u8 = DISPATCH_KEY_ZERO;

pub const DISPATCH_KEY_PRIVATEUSE3:                 u8 = 27;
pub const DISPATCH_KEY_COMPOSITE_EXPLICIT_AUTOGRAD: u8 = 59;
pub const DISPATCH_KEY_CPU:                         u8 = 1;
pub const DISPATCH_KEY_CUDA:                        u8 = 2;
pub const DISPATCH_KEY_AUTOGRAD_PRIVATEUSE1:        u8 = 42;
pub const DISPATCH_KEY_AUTOGRAD_PRIVATEUSE2:        u8 = 43;
pub const DISPATCH_KEY_AUTOGRAD_PRIVATEUSE3:        u8 = 44;
pub const DISPATCH_KEY_AUTOCAST_CUDA:               u8 = 47;

/// it would be better if this were a constant assertion... but oh well
#[test] fn test_dispatch_key_values() {

    /// the enum values are from c++ and a bit ad-hoc
    ///
    /// we will put some extra checks here just to ensure things stay the same
    ///
    assert_eq!{DISPATCH_KEY_PRIVATEUSE3,                  DispatchKey::PrivateUse3.into()}
    assert_eq!{DISPATCH_KEY_COMPOSITE_EXPLICIT_AUTOGRAD,  DispatchKey::CompositeExplicitAutograd.into()}
    assert_eq!{DISPATCH_KEY_CPU,                          DispatchKey::CPU.into()}
    assert_eq!{DISPATCH_KEY_CUDA,                         DispatchKey::Cuda.into()}
    assert_eq!{DISPATCH_KEY_AUTOGRAD_PRIVATEUSE1,         DispatchKey::AutogradPrivateUse1.into()}
    assert_eq!{DISPATCH_KEY_AUTOGRAD_PRIVATEUSE2,         DispatchKey::AutogradPrivateUse2.into()}
    assert_eq!{DISPATCH_KEY_AUTOGRAD_PRIVATEUSE3,         DispatchKey::AutogradPrivateUse3.into()}
    assert_eq!{DISPATCH_KEY_AUTOCAST_CUDA,                DispatchKey::AutocastCUDA.into()}

    assert_eq!{DispatchKey::CPUTensorId.into(),               DISPATCH_KEY_CPU}
    assert_eq!{DispatchKey::CUDATensorId.into(),              DISPATCH_KEY_CUDA}
    assert_eq!{DispatchKey::DefaultBackend.into(),            DISPATCH_KEY_COMPOSITE_EXPLICIT_AUTOGRAD}
    assert_eq!{DispatchKey::PrivateUse1_PreAutograd.into(),   DISPATCH_KEY_AUTOGRAD_PRIVATEUSE1}
    assert_eq!{DispatchKey::PrivateUse2_PreAutograd.into(),   DISPATCH_KEY_AUTOGRAD_PRIVATEUSE2}
    assert_eq!{DispatchKey::PrivateUse3_PreAutograd.into(),   DISPATCH_KEY_AUTOGRAD_PRIVATEUSE3}
    assert_eq!{DispatchKey::Autocast.into(),                  DISPATCH_KEY_AUTOCAST_CUDA}

    assert_eq!{DispatchKey::EndOfBackendKeys.into(),   DispatchKey::PrivateUse3.into()}
    assert_eq!{DispatchKey::EndOfAliasKeys.into(),     DispatchKey::CompositeExplicitAutograd.into()}
}

impl Into<u8> for DispatchKey {

    fn into(self) -> u8 {
        self.c_protocol_number()
    }
}

/**
  | Semantically, a dispatch key identifies
  | a possible "level" in our dispatch, for which
  | a handler may be registered.
  |
  | Traditional backends like CPU and Cuda get
  | dispatch keys; however, so do "wrapping" layers
  | like Variable (for autograd handling).
  |
  | In implementation terms, the dispatch key
  | identifies a specific "bit" in
  | a DispatchKeySet. Higher bit indexes get
  | handled by dispatching first (because we "count
  | leading zeros" when we extract the highest
  | priority dispatch key.)
  |
  | NOTE: Keep the list in sync with `DispatchKey`
  | in tools/codegen/model.py
  |
  */
#[derive(Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub enum DispatchKey {

    /**
      | ~~~~~~~UNDEFINED ~~~~~~~~//
      |
      | This is not a "real" tensor id, but it exists
      | to give us a "nullopt" element we can return
      | for cases when a DispatchKeySet contains no
      | elements.
      |
      | You can think a more semantically accurate
      | definition of DispatchKey is:
      |
      |    using DispatchKey = optional<RealDispatchKey>
      |
      | and Undefined == nullopt.
      |
      | We didn't actually represent it this way
      | because optional<RealDispatchKey> would take
      | two words, when DispatchKey fits in eight
      | bits.
      */
    Undefined,

    /**
      | Define an alias for Undefined to represent
      | CatchAll (long term this will get eliminated,
      | but for now it's convenient)
      |
      */
    CatchAll,

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // A "backend" is colloquially used to refer to handlers for dispatch
    // which actually implement the numerics of an operation in question.
    //
    // Due to the nature of the enum, these backends are specified in
    // an ordered way, but for most backends this order is not semantically
    // meaningful (e.g., it's valid to reorder these backends without changing
    // semantics).  The only situation when backend ordering is meaningful
    // is when the backend participates in multiple dispatch with another
    // backend; e.g., CPU and SparseCPU (sparse must have
    // higher priority).

    // Here are backends which you think of as traditionally specifying
    // how to implement operations on some device.


    /// registered at build/aten/src/ATen/RegisterCPU.cpp
    CPU, 

    /// registered at build/aten/src/ATen/RegisterCUDA.cpp
    Cuda, 

    /// NB: I think this is not actually used, due to Note [Masquerading as Cuda]
    HIP, 

    /// Xilinx support lives out of tree at https://gitlab.com/pytorch-complex/vitis_kernels
    FPGA, 

    /// unused externally, but tested at test/cpp_extensions/msnpu_extension.cpp
    MSNPU, 

    /// lives out of tree at https://github.com/pytorch/xla
    XLA, 

    /// lives out of tree at https://github.com/pytorch/MLCompute
    MLC, 

    Vulkan,
    Metal,

    /// For out of tree Intel's heterogeneous computing plug-in
    XPU, 

    /// For out of tree & closed source integration of HPU / Habana
    HPU, 


    // A meta tensor is a tensor without any data associated with it.  (They
    // have also colloquially been referred to as tensors on the "null" device).
    // A meta tensor can be used to dry run operators without actually doing any
    // computation, e.g., add on two meta tensors would give you another meta
    // tensor with the output shape and dtype, but wouldn't actually add anything.
    Meta,

    // Here are backends which specify more specialized operators
    // based on the dtype of the tensor.

    ///  registered at build/aten/src/ATen/RegisterQuantizedCPU.cpp
    QuantizedCPU, 

    /// registered at build/aten/src/ATen/RegisterQuantizedCUDA.cpp
    QuantizedCUDA, 

    /// For out of tree Intel's heterogeneous computing plug-in
    QuantizedXPU, 

    // This backend is to support custom RNGs; it lets you go
    // to a different kernel if you pass in a generator that is not a
    // traditional CPUGeneratorImpl/CUDAGeneratorImpl.  To make use of this
    // key:
    //  1) set it as a second parameter of Generator constructor call in
    //     the user-defined PRNG class.
    //  2) use it as a dispatch key while registering custom kernels
    //     (templatized kernels specialized for user-defined PRNG class)
    // intended for out of tree use; tested by aten/src/ATen/test/rng_test.cpp
    CustomRNGKeyId,

    // Here are backends which specify more specialized operators
    // based on the layout of the tensor.  Note that the sparse backends
    // are one case where ordering matters: sparse multi-dispatches with
    // the corresponding dense tensors, and must be handled before them.

    /// registered at build/aten/src/ATen/RegisterMkldnnCPU.cpp 
    /// NB: not to be confused with MKLDNN, which is Caffe2 only
    MkldnnCPU, 

    /// registered at build/aten/src/ATen/RegisterSparseCPU.cpp
    SparseCPU, 

    /// registered at build/aten/src/ATen/RegisterSparseCUDA.cpp
    SparseCUDA, 

    /// TODO: I think this is not actually used, due to Note [Masquerading as Cuda]
    SparseHIP, 

    /// For out of tree Intel's heterogeneous computing plug-in
    SparseXPU, 


    SparseCsrCPU,
    SparseCsrCUDA,

    /// lives out of tree at https://github.com/pytorch/nestedtensor
    NestedTensor, 
                  
    /// Here are reserved backends for user-defined backends, see Note [Private use DispatchKey]
    ///
    /// To see some example about how to use this, check out MSNPU
    ///
    PrivateUse1,
    PrivateUse2,
    PrivateUse3,

    /// Define an alias key to represent end of backend dispatch keys.
    ///
    /// If you add new backend keys after PrivateUse3, please also update it here.
    ///
    EndOfBackendKeys,

    /// In some situations, it is not immediately obvious what the correct backend for function is,
    /// because the function in question doesn't have any "tensor" arguments.  
    ///
    /// In this case, a BackendSelect function can be registered to implement the custom
    /// determination of the correct backend.
    ///
    BackendSelect,

    /// See Note [Out-of-tree vmap+grad prototype]
    FuncTorchPython, 

    // The named dispatch key is set for any tensors with named dimensions.
    // Although we have a dispatch key for named tensors, for historical reasons,
    // this dispatch key doesn't do any of the substantive functionality for named
    // tensor (though, hypothetically, it could!)  At the moment, it's just
    // responsible for letting us give good error messages when operations
    // don't support named tensors.
    //
    // NB: If you ever consider moving named tensor functionality into
    // this dispatch key, note that it might be necessary add another dispatch
    // key that triggers before composite operators, in case a composite operator
    // has named dimension propagation that doesn't match that of its
    // constituent parts.
    Named,

    // The Conjugate dispatch key is set for any tensors that need to perform
    // conjugation
    // This is implemented at a dispatch level right before any backends run
    Conjugate,

    // See Note [Out-of-tree vmap+grad prototype]. The purpose of this key
    // is to insert code after the "autograd subsystem" runs, so this key should
    // be directly after ADInplaceOrView and all of the autograd keys.
    FuncTorchDynamicLayerBackMode,

    // Note [ADInplaceOrView key]
    // ADInplaceOrView key is used by inplace or view ops to register a kernel
    // that does additional setup for future autograd computation.
    //
    // 1. For inplace ops this kernel does version bump
    // 2. For view ops this kernel does `as_view` setup where we properly setup
    //    DifferentiableViewMeta on the view tensors.
    //
    // For other ops it's fallthrough kernel since there's no extra
    // work to do.
    //
    // Note [Dream: skip VariableType kernel when requires_grad=false]
    //
    // In an ideal world where we can skip VariableType kernel for inputs
    // with requires_grad=false, instead of a fallthrough kernel, we'll
    // register a kernel shown below to all functional ops as well:
    // TorchTensor my_functional_op(...) {
    //   {
    //     // Note for every op in VariableType, you need to go through
    //     // `AutoDispatchBelowADInplaceOrView` guard exactly once to add the
    //     // key to TLS excluded set. If you don't go through it at all,
    //     // inplace/view ops called through `` inside your backend
    //     // kernel will dispatch to ADInplaceOrView kernels and do a lot
    //     // of extra work.
    //     AutoDispatchBelowADInplaceOrView guard;
    //     redispatch::my_functional_op(...);
    //   }
    // }
    // But this work is currently blocked since it adds an extra dispatch
    // for all ops and it's non-trivial overhead at model level(a few percents).
    // Thus our current approach takes advantage of the fact every kernel go
    // through VariableType kernel first and pulls the
    // `AutoDispatchBelowADInplaceOrView` guard of functional ops
    // up to the `VariableType` kernel. Thus we only add the extra dispatch
    // to view/inplace ops to minimize its perf impact to real models.
    ADInplaceOrView,

    // Note [Alias Dispatch Key : Autograd]
    // All backends are oblivious to autograd; autograd is handled as a
    // layer which happens on top of all backends. It inspects the autograd
    // metadata of all inputs, determines what autograd metadata should be
    // constructed by the output, and otherwise defers to the backend to
    // actually do the numeric computation.  Autograd contains
    // the bulk of this logic.

    // Autograd is now an alias dispatch key which by default maps to all
    // backend-specific autograd keys.
    // Backend-specific allow backends to override the default kernel registered
    // to Autograd key as needed.
    // For example, XLA wants to define autograd for einsum directly.
    // Registering a custom autograd implementation at the XLA key won't work
    // because we process Autograd before XLA.  This key has higher priority and
    // gets processed first.  You generally should NOT redispatch after handling
    // autograd here (since that would result in execution of the Autograd
    // operator, which you're trying to skip).  In AutogradXLA implementations,
    // you are responsible for handling autograd yourself, or deferring to other
    // operators which support autograd.

    // Currently we only have backend-specific autograd keys for CPU/Cuda/XLA and
    // reserved user-defined backends. All other in-tree backends share the
    // AutogradOther key. We can add specific autograd key for those backends
    // upon request.
    AutogradOther,
    AutogradCPU,
    AutogradCUDA,
    AutogradXLA,
    AutogradXPU,
    AutogradMLC,
    AutogradHPU,

    /// lives out of tree at https://github.com/pytorch/nestedtensor
    ///
    AutogradNestedTensor, 

    // Here are some reserved pre-autograd keys for user-defined backends, see
    // Note [Private use DispatchKey]
    AutogradPrivateUse1,
    AutogradPrivateUse2,
    AutogradPrivateUse3,

    Tracer,

    // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
    // and inputs are saved for backward in the post-autocast type.
    AutocastCPU,
    AutocastCUDA,

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // There are a number of alternative modes which may want to handle before
    // autograd; for example, error checking, tracing, profiling or vmap.  They
    // go here.

    /// See Note [Out-of-tree vmap+grad prototype]
    FuncTorchBatched, 

    /// See Note [Out-of-tree vmap+grad prototype]
    FuncTorchVmapMode, 

    // This is the dispatch key for BatchedTensorImpl, which is used to implement
    // batching rules for vmap.
    Batched,

    // When we are inside a vmap, all tensors dispatch on this key.
    // See Note: [DispatchKey::VmapMode usage] for more details.
    VmapMode,

    /// See Note [Out-of-tree vmap+grad prototype]
    FuncTorchGradWrapper, 

    /// See Note [Out-of-tree vmap+grad prototype]
    FuncTorchDynamicLayerFrontMode, 

    // TESTING: This is intended to be a generic testing tensor type id.
    // Don't use it for anything real; its only acceptable use is within a single
    // process test.  Use it by creating a TensorImpl with this DispatchKey, and
    // then registering operators to operate on this type id.  See
    // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
    TESTING_ONLY_GenericWrapper,

    // TESTING: This is intended to be a generic testing tensor type id.
    // Don't use it for anything real; its only acceptable use is within a ingle
    // process test.  Use it by toggling the mode on and off via
    // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
    // to operate on this type id.  See
    // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
    // for a usage example
    TESTING_ONLY_GenericMode,

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    /// Sentinel, end of runtime keys.
    NumDispatchKeys, 

    // ~~~~~~~~~~~~~~~~~~~~~~ Alias Dispatch Keys ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Alias dispatch keys are synthetic dispatch keys which map to multiple
    // runtime dispatch keys. Alisa keys have precedence, but they are always
    // lower precedence than runtime keys. You can register a kernel to an
    // alias key, the kernel might be populated to the mapped runtime keys
    // during dispatch table computation.
    // If a runtime dispatch key has multiple kernels from alias keys, which
    // kernel wins is done based on the precedence of alias keys (but runtime
    // keys always have precedence over alias keys).
    // Alias keys won't be directly called during runtime.

    // See Note [Alias Dispatch Key : Autograd]
    Autograd,

    /// registered at build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
    CompositeImplicitAutograd, 

    /// registered at build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
    CompositeExplicitAutograd, 

    // Define an alias key to represent end of alias dispatch keys.
    // If you add new alias keys after Autograd, please also update it here.
    EndOfAliasKeys,

    // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // The aliases exist for backwards compatibility reasons, they shouldn't
    // be used
    CPUTensorId,
    CUDATensorId,
    DefaultBackend,
    PrivateUse1_PreAutograd,
    PrivateUse2_PreAutograd,
    PrivateUse3_PreAutograd,
    Autocast,
}

impl DispatchKey {
    pub fn c_protocol_number(&self) -> u8 {
        match self {
            DispatchKey::Undefined                      => DISPATCH_KEY_UNDEFINED,
            DispatchKey::CatchAll                       => DISPATCH_KEY_CATCHALL,
            DispatchKey::CPU                            => 1,
            DispatchKey::Cuda                           => 2,
            DispatchKey::HIP                            => 3,
            DispatchKey::FPGA                           => 4,
            DispatchKey::MSNPU                          => 5,
            DispatchKey::XLA                            => 6,
            DispatchKey::MLC                            => 7,
            DispatchKey::Vulkan                         => 8,
            DispatchKey::Metal                          => 9,
            DispatchKey::XPU                            => 10,
            DispatchKey::HPU                            => 11,
            DispatchKey::Meta                           => 12,
            DispatchKey::QuantizedCPU                   => 13,
            DispatchKey::QuantizedCUDA                  => 14,
            DispatchKey::QuantizedXPU                   => 15,
            DispatchKey::CustomRNGKeyId                 => 16,
            DispatchKey::MkldnnCPU                      => 17,
            DispatchKey::SparseCPU                      => 18,
            DispatchKey::SparseCUDA                     => 19,
            DispatchKey::SparseHIP                      => 20,
            DispatchKey::SparseXPU                      => 21,
            DispatchKey::SparseCsrCPU                   => 22,
            DispatchKey::SparseCsrCUDA                  => 23,
            DispatchKey::NestedTensor                   => 24,
            DispatchKey::PrivateUse1                    => 25,
            DispatchKey::PrivateUse2                    => 26,
            DispatchKey::PrivateUse3                    => 27,
            DispatchKey::EndOfBackendKeys               => DISPATCH_KEY_PRIVATEUSE3,
            DispatchKey::BackendSelect                  => 28,
            DispatchKey::FuncTorchPython                => 29,
            DispatchKey::Named                          => 30,
            DispatchKey::Conjugate                      => 31,
            DispatchKey::FuncTorchDynamicLayerBackMode  => 32,
            DispatchKey::ADInplaceOrView                => 33,
            DispatchKey::AutogradOther                  => 34,
            DispatchKey::AutogradCPU                    => 35,
            DispatchKey::AutogradCUDA                   => 36,
            DispatchKey::AutogradXLA                    => 37,
            DispatchKey::AutogradXPU                    => 38,
            DispatchKey::AutogradMLC                    => 39,
            DispatchKey::AutogradHPU                    => 40,
            DispatchKey::AutogradNestedTensor           => 41,
            DispatchKey::AutogradPrivateUse1            => 42,
            DispatchKey::AutogradPrivateUse2            => 43,
            DispatchKey::AutogradPrivateUse3            => 44,
            DispatchKey::Tracer                         => 45,
            DispatchKey::AutocastCPU                    => 46,
            DispatchKey::AutocastCUDA                   => 47,
            DispatchKey::FuncTorchBatched               => 48,
            DispatchKey::FuncTorchVmapMode              => 49,
            DispatchKey::Batched                        => 50,
            DispatchKey::VmapMode                       => 51,
            DispatchKey::FuncTorchGradWrapper           => 52,
            DispatchKey::FuncTorchDynamicLayerFrontMode => 53,
            DispatchKey::TESTING_ONLY_GenericWrapper    => 54,
            DispatchKey::TESTING_ONLY_GenericMode       => 55,
            DispatchKey::NumDispatchKeys                => 56,
            DispatchKey::Autograd                       => 57,
            DispatchKey::CompositeImplicitAutograd      => 58,
            DispatchKey::CompositeExplicitAutograd      => 59,
            DispatchKey::EndOfAliasKeys                 => DISPATCH_KEY_COMPOSITE_EXPLICIT_AUTOGRAD,
            DispatchKey::CPUTensorId                    => DISPATCH_KEY_CPU,
            DispatchKey::CUDATensorId                   => DISPATCH_KEY_CUDA,
            DispatchKey::DefaultBackend                 => DISPATCH_KEY_COMPOSITE_EXPLICIT_AUTOGRAD,
            DispatchKey::PrivateUse1_PreAutograd        => DISPATCH_KEY_AUTOGRAD_PRIVATEUSE1,
            DispatchKey::PrivateUse2_PreAutograd        => DISPATCH_KEY_AUTOGRAD_PRIVATEUSE2,
            DispatchKey::PrivateUse3_PreAutograd        => DISPATCH_KEY_AUTOGRAD_PRIVATEUSE3,
            DispatchKey::Autocast                       => DISPATCH_KEY_AUTOCAST_CUDA,
        }
    }
}

/*
  | Note [Private use DispatchKey]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | Private use tensor IDs are preallocated tensor
  | type IDs for use in user applications.  Similar
  | to private use fields in HTTP, they can be used
  | by end users for experimental or private
  | applications, without needing to "standardize"
  | the tensor ID (which would be done by
  | submitting a PR to PyTorch to add your type
  | ID).
  |
  | Private use tensor IDs are appropriate to use
  | if you want to experiment with adding a new
  | tensor type (without having to patch PyTorch
  | first) or have a private, non-distributed
  | application that needs to make use of a new
  | tensor type.  Private use tensor IDs are NOT
  | appropriate to use for libraries intended to be
  | distributed to further users: please contact
  | the PyTorch developers to get a type ID
  | registered in this case.
  |
  | We provide two classes of private user tensor
  | id: regular DispatchKeys and Autograd
  | DispatchKeys.  DispatchKeys serve the role of
  | ordinary "backend" DispatchKeys; if you were
  | adding support for a new type of accelerator,
  | you would use a backend DispatchKey, and
  | ideally automatically reuse AutogradOther
  | definitions already defined in PyTorch.
  | AutogradPrivateUse DispatchKeys serve as
  | "wrapper" DispatchKeys: they are only necessary
  | for tensors that compose multiple internal
  | tensors, and for cases when the built-in
  | autograd formulas for operators are not
  | appropriate.
  */

/**
  | DispatchKey is used as index into 64-bit
  | bitmask; you must have less than 64 entries
  |
  */
const_assert!{
    (DispatchKey::NumDispatchKeys as u8) < 64
}

/**
  | These are some convenience identifiers for
  | dispatch keys which are shorter to type than
  | their long counterparts.
  |
  | Note that some of these dispatch keys directly
  | correspond to DeviceType; and most APIs that
  | accept DispatchKey also accept DeviceType;
  | e.g., Torchdispatch(TorchkCPU, ...) is also
  | valid.
  |
  */
pub const K_AUTOGRAD: DispatchKey = DispatchKey::Autograd;

/**
  | Check if a DispatchKey is an alias mapping
  | to other runtime keys.
  |
  */
#[inline] pub fn is_alias_dispatch_key(k: DispatchKey) -> bool {
    
    todo!();
        /*
            return k > DispatchKey::NumDispatchKeys && k <= DispatchKey::EndOfAliasKeys;
        */
}

/**
  | NB: You really shouldn't use this instance;
  | this enum is guaranteed to be pretty small so
  | a regular array should be acceptable.
  |
  */
impl Hash for DispatchKey {

    fn hash<H>(&self, state: &mut H) where H: Hasher 
    {
        
        todo!();
        /*
            return static_cast<size_t>(x);
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/DispatchKey.cpp]

pub fn to_string(t: DispatchKey) -> *const u8 {
    
    todo!();
        /*
            switch (t) {
        case DispatchKey::Undefined:
          return "Undefined";

        case DispatchKey::CPU:
          return "CPU";
        case DispatchKey::Cuda:
          return "Cuda";

        case DispatchKey::HIP:
          return "HIP";
        case DispatchKey::FPGA:
          return "FPGA";
        case DispatchKey::XPU:
          return "XPU";
        case DispatchKey::MSNPU:
          return "MSNPU";
        case DispatchKey::XLA:
          return "XLA";
        case DispatchKey::MLC:
          return "MLC";
        case DispatchKey::HPU:
          return "HPU";
        case DispatchKey::Vulkan:
          return "Vulkan";
        case DispatchKey::Metal:
          return "Metal";
        case DispatchKey::QuantizedCPU:
          return "QuantizedCPU";
        case DispatchKey::QuantizedCUDA:
          return "QuantizedCUDA";
        case DispatchKey::QuantizedXPU:
          return "QuantizedXPU";

        case DispatchKey::CustomRNGKeyId:
          return "CustomRNGKeyId";

        case DispatchKey::MkldnnCPU:
          return "MkldnnCPU";
        case DispatchKey::SparseCPU:
          return "SparseCPU";
        case DispatchKey::SparseCUDA:
          return "SparseCUDA";
        case DispatchKey::SparseCsrCPU:
          return "SparseCsrCPU";
        case DispatchKey::SparseCsrCUDA:
          return "SparseCsrCUDA";
        case DispatchKey::SparseHIP:
          return "SparseHIP";
        case DispatchKey::SparseXPU:
          return "SparseXPU";

        case DispatchKey::NestedTensor:
          return "NestedTensor";

        case DispatchKey::PrivateUse1:
          return "PrivateUse1";
        case DispatchKey::PrivateUse2:
          return "PrivateUse2";
        case DispatchKey::PrivateUse3:
          return "PrivateUse3";

        case DispatchKey::Conjugate:
          return "Conjugate";
        case DispatchKey::Meta:
          return "Meta";

        case DispatchKey::ADInplaceOrView:
          return "ADInplaceOrView";

        case DispatchKey::Autograd:
          return "Autograd";
        case DispatchKey::AutogradCPU:
          return "AutogradCPU";
        case DispatchKey::AutogradCUDA:
          return "AutogradCUDA";
        case DispatchKey::AutogradXLA:
          return "AutogradXLA";
        case DispatchKey::AutogradMLC:
          return "AutogradMLC";
        case DispatchKey::AutogradHPU:
          return "AutogradHPU";
        case DispatchKey::AutogradNestedTensor:
          return "AutogradNestedTensor";
        case DispatchKey::AutogradPrivateUse1:
          return "AutogradPrivateUse1";
        case DispatchKey::AutogradPrivateUse2:
          return "AutogradPrivateUse2";
        case DispatchKey::AutogradPrivateUse3:
          return "AutogradPrivateUse3";
        case DispatchKey::AutogradOther:
          return "AutogradOther";
        case DispatchKey::BackendSelect:
          return "BackendSelect";
        case DispatchKey::Named:
          return "Named";

        case DispatchKey::Tracer:
          return "Tracer";

        case DispatchKey::Autocast:
          return "Autocast";

        case DispatchKey::Batched:
          return "Batched";

        case DispatchKey::VmapMode:
          return "VmapMode";

        case DispatchKey::CompositeImplicitAutograd:
          return "CompositeImplicitAutograd";

        case DispatchKey::CompositeExplicitAutograd:
          return "CompositeExplicitAutograd";

        case DispatchKey::TESTING_ONLY_GenericWrapper:
          return "TESTING_ONLY_GenericWrapper";

        case DispatchKey::TESTING_ONLY_GenericMode:
          return "TESTING_ONLY_GenericMode";

        // Note [Out-of-tree vmap+grad prototype]
        // The following keys are used in the implementation of the out-of-tree
        // composable functions transforms (vmap+grad) prototype that lives at
        // https://github.com/zou3519/functorch
        // We plan on eventually upstreaming the prototype into core, at which
        // point it will have a different design that should use fewer keys.
        case DispatchKey::FuncTorchPython:
          return "FuncTorchPython";
        case DispatchKey::FuncTorchDynamicLayerBackMode:
          return "FuncTorchDynamicLayerBackMode";
        case DispatchKey::FuncTorchDynamicLayerFrontMode:
          return "FuncTorchDynamicLayerFrontMode";
        case DispatchKey::FuncTorchGradWrapper:
          return "FuncTorchGradWrapper";
        case DispatchKey::FuncTorchVmapMode:
          return "FuncTorchVmapMode";
        case DispatchKey::FuncTorchBatched:
          return "FuncTorchBatched";

        default:
          return "UNKNOWN_TENSOR_TYPE_ID";
      }
        */
}

impl fmt::Display for DispatchKey {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            return str << toString(rhs);
        */
    }
}

/**
  | for a given backend key, return the associated
  | autograd key.
  |
  | for non-backend keys, return AutogradOther as
  | a default.
  |
  | Note: it's convenient and fast to return
  | a default here rather than (say) returning an
  | optional<DispatchKey>, or throwing.
  |
  | But it makes callers responsible for either a)
  | enforcing the invariant that only backend keys
  | be passed as arguments, or b) interpreting our
  | return value carefully.
  |
  */
pub fn get_autograd_key_from_backend(t: DispatchKey) -> DispatchKey {
    
    todo!();
        /*
            switch (t) {
        case DispatchKey::CPU:
          return DispatchKey::AutogradCPU;
        case DispatchKey::XPU:
          return DispatchKey::AutogradXPU;
        case DispatchKey::Cuda:
          return DispatchKey::AutogradCUDA;
        case DispatchKey::XLA:
          return DispatchKey::AutogradXLA;
        case DispatchKey::MLC:
          return DispatchKey::AutogradMLC;
        case DispatchKey::HPU:
          return DispatchKey::AutogradHPU;
        case DispatchKey::NestedTensor:
          return DispatchKey::AutogradNestedTensor;
        case DispatchKey::PrivateUse1:
          return DispatchKey::AutogradPrivateUse1;
        case DispatchKey::PrivateUse2:
          return DispatchKey::AutogradPrivateUse2;
        case DispatchKey::PrivateUse3:
          return DispatchKey::AutogradPrivateUse3;
        default:
          return DispatchKey::AutogradOther;
      }
        */
}
