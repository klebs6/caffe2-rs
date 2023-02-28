/*!
  | TODO This whole file should be deleted
  | and replaced with the mechanism described
  | in https://github.com/pytorch/pytorch/issues/29548
  |
  | -------------
  |
  | This file implements a variable fallback
  | kernel for custom operators.
  | 
  | Since tensors always have the Autograd
  | set, but custom operators usually don't
  | have a kernel registered for Autograd,
  | the dispatcher will call into this fallback
  | kernel instead.
  | 
  | -----------
  | @note
  | 
  | this is not a correct autograd implementation.
  | It will just fallthrough to the custom
  | operator implementation.
  | 
  | If you want a custom operator to work
  | with autograd, you need to use autograd::Function
  | so that the custom operator implementation
  | knows how to do autograd.
  | 
  | Note also that ops from native_functions.yaml
  | register their own variable kernels,
  | so this is never called for them.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp]

/**
  | Register fallthrough for Autograd backends
  | dispatch keys
  |
  | NB: But not the private use ones; maybe the
  | extension wants to override it themselves!
  */
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(_, AutogradOther, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(_, AutogradCPU, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(_, AutogradXPU, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(_, AutogradCUDA, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    TORCH_LIBRARY_IMPL(_, AutogradMLC, m) {
      m.fallback(TorchCppFunction::makeFallthrough());
    }

    // see Note [ADInplaceOrView key]
    TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
          m.fallback(TorchCppFunction::makeFallthrough());
    }
    */
}
