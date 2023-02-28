/*!
  | The legacy mechanism for dispatching operators
  | in ATen is a Type object, which is essentially
  | a giant virtual dispatch table for every
  | operation we support dynamically dispatching
  | over.
  |
  | This has been deprecated in favor of
  | ATenDispatch, and in the future, c10
  | dispatcher.
  |
  | TODO: Clean up what remains here
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/LegacyTypeDispatch.h]

/*
  | A RAII, thread local (!) guard that will
  | disable dispatch to variable handler.
  |
  | NOTE [ Treating Variables as non-Variables in
  | type dispatch ]
  |
  | What exactly does AutoDispatchBelowAutograd do?
  | The short answer is, it causes dispatches on
  | ATen functions to go to the non-variable
  | implementation, bypassing autograd handling
  | (and also profiling and tracing).
  |
  | To understand why this guard exists, it's
  | helpful to understand the history behind how
  | Variable was implemented.  Previously,
  | Variables were implemented as a wrapper on
  | Tensors; so the act of processing a Variable
  | involved unwrapping the underlying Tensor, and
  | then calling the underlying base operation on
  | /that/ operation
  |
  | However, after the Variable/Tensor merge, there
  | is no concept of unwrapping a tensor anymore.
  | If you just call the operation on the same
  | variable again inside your VariableType
  | handler, you'll dispatch back to VariableType,
  | which is not what we want.
  |
  | The solution to the above problem is to add
  | `AutoDispatchBelowAutograd`, which when enabled
  | will cause `legacyTensorType()` and `getType()`
  | to always return non-Variable type, even if the
  | tensor being called on is a variable.
  */

/* 
 | Note [AutoDispatchBelowAutograd]
 |
 | AutoDispatchBelowAutograd is **INTERNAL ONLY**
 | that it should be used for kernel
 | implementations and customized C++ kernels.
 |
 | If you are looking for a guard to run workload
 | in inference mode, please use InferenceMode
 | RAII which is user facing API.
 |
 | In the past AutoDispatchBelowAutograd(or its
 | old version AutoNonVariableTypeMode) was used
 | in the user code for inference-only workload,
 | this was under risk of producing wrong results
 | silently in some edge cases. For example:
 |
 | ```
 |  TorchTensor s = Torchones({1, 2, 3}).set_requires_grad(true);
 |  TorchTensor out = s * s;
 |  {
 |    AutoDispatchBelowAutograd guard;
 |    s.add_(1);  // Skips version bump on `s`.
 |  }
 |  // WRONG GRADIENT! s.grad() are now computed using `s` value after the
 |  // inplace update.
 |  out.backward(Torchones_like(out));
 | ```
 | Users should use `InferenceMode` here so that
 | it'll properly throw an error saying "one of
 | the variables needed for gradient computation
 | has be modified."
 |
 */
pub struct AutoDispatchBelowAutograd {

    /**
      | disable all autograd dispatch keys
      |
      */
    autograd_guard: ExcludeDispatchKeyGuard,
}

impl Default for AutoDispatchBelowAutograd {
    
    fn default() -> Self {
        todo!();
        /*
        : autograd_guard(autograd_dispatch_keyset),

        
        */
    }
}

/// TODO: AutoNonVariableTypeMode should be
/// removed in release 1.10.
///
pub struct AutoNonVariableTypeMode {

    /**
      | disable all autograd dispatch keys
      |
      */
    autograd_guard: ExcludeDispatchKeyGuard,
}

impl AutoNonVariableTypeMode {
    
    pub fn new(enabled: bool) -> Self {
        let enabled: bool = enabled.unwrap_or(true);
        todo!();
        /*
        : autograd_guard(autograd_dispatch_keyset),

            TORCH_WARN_ONCE("AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. "
            "For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, "
            "If you are looking for a user facing API to enable running your inference-only "
            "workload, please use InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code "
            "is under risk of producing silent wrong result in some edge cases. "
            "See Note [AutoDispatchBelowAutograd] for more details.");
        TORCH_INTERNAL_ASSERT(enabled);
        */
    }
}



/* Note [AutoDispatchBelowADInplaceOrView]
 *
 * AutoDispatchBelowADInplaceOrView is equivalent
 * to AutoNonVariableTypeMode before we split
 * inplace & view ops out of VariableType kernel.
 *
 * Note this guard is used in VariableType kernels
 * for functional ops as well as ADInplaceOrView
 * kernels for inplace/view ops to enforce the
 *
 * Invariant:
 *
 *   Once you are in VariableType/ADInplaceOrView
 *   kernel for an op, you never go back to
 *   a kernel on same dispatch key until you
 *   finish the current op.
 */
pub struct AutoDispatchBelowADInplaceOrView {

    /**
      | disable Autograd & ADInplaceOrView
      | dispatch keys
      |
      */
    dispatch_key_guard: ExcludeDispatchKeyGuard,
}

impl Default for AutoDispatchBelowADInplaceOrView {
    
    fn default() -> Self {
        todo!();
        /*
        : dispatch_key_guard(autograd_dispatch_keyset_with_ADInplaceOrView),

        
        */
    }
}
