crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/InferenceMode.h]

/**
  | A RAII, thread local (!) guard that enables or
  | disables inference mode upon construction, and
  | sets it back to the original value upon
  | destruction.
  |
  */
pub struct InferenceMode {
    prev_mode:   bool,
    prev_keyset: LocalDispatchKeySet,
    grad_mode:   AutoGradMode,
}

impl Drop for InferenceMode {

    fn drop(&mut self) {
        todo!();
        /*
            _set_enabled(prev_mode);
        _force_tls_local_dispatch_key_set(prev_keyset);
        */
    }
}

impl InferenceMode {

    /**
      | Note [Expected TLS state in InferenceMode]:
      |
      |   InferenceMode: ADInplaceOrView not in
      |   raw_local_dispatch_key_set.included(),
      |                  Autograd in raw_local_dispatch_key_set.excluded()
      |                  GradMode is disabled.
      |   NormalMode: ADInplaceOrView in raw_local_dispatch_key_set.included(),
      |               Autograd not in raw_local_dispatch_key_set.excluded()
      |               GradMode is enabled by default unless toggled manually
      |               through other APIs, e.g. NoGradGuard.
      |
      | Invariant:
      |
      | - ADInplaceOrView is never in the excluded set
      |
      | - Autograd is never in the included set
      |
      | - Setting InferenceMode will set GradMode
      | accordingly, but not vice versa.
      |
      |
      |  1. Why do we put ADInplaceOrView in included
      |  set outside InferenceMode?
      |
      |     Inplace update to inference tensor
      |     outside InferenceMode is not allowed.
      |
      |     See Note [Inplace update inference tensor]
      |     for more details.
      |
      |     Without going through ADInplaceOrView
      |       kernel, we cannot throw error for
      |       `inference_tensor.add_(1)` case.
      |
      | 2. Why not put ADInplaceOrView in the excluded
      |  set inside InferenceMode?
      |
      |    For example:
      |    TorchTensor a = Torchones({1, 2, 3}).set_requires_grad(true);
      |    TorchTensor k = a + 2;
      |    {
      |      InferenceMode guard(true);
      |      k.add_(2);
      |    }
      |    `k.add_(2)` still need to go through
      |    ADInplaceOrView kernel so that it's prepared
      |    for future autograd.
      |
      | 3. Why does setting InferenceMode also set GradMode?
      |
      |    This is required since InferenceMode is
      |    a faster and more restricive version of
      |    NoGradGuard. All runtime checks using
      |    GradMode::is_enabled() are applicable to
      |    InferenceMode as well,
      |    e.g. `tensorTypeInCurrentExecutionContext`
      |    in interpreter.cpp.
      |
      */
    pub fn new(enabled: Option<bool>) -> Self {

        let enabled: bool = enabled.unwrap_or(true);

        todo!();
        /*


            : prev_mode(InferenceMode::is_enabled()),
            prev_keyset(tls_local_dispatch_key_set()),
            grad_mode(AutoGradMode(!enabled)) 

        _set_enabled(enabled);
        DispatchKeySet included = enabled
            ? prev_keyset.included_.remove(DispatchKey::ADInplaceOrView)
            : prev_keyset.included_.add(DispatchKey::ADInplaceOrView);
        DispatchKeySet excluded = enabled
            ? (prev_keyset.excluded_ | autograd_dispatch_keyset)
            : (prev_keyset.excluded_ - autograd_dispatch_keyset);
        PODLocalDispatchKeySet cur_keyset;
        cur_keyset.set_included(included);
        cur_keyset.set_excluded(excluded);
        _force_tls_local_dispatch_key_set(cur_keyset);
        */
    }
    
    /**
      | Invariant:
      |   is_enabled() ==
      |   !tls_is_dispatch_key_included(DispatchKey::ADInplaceOrView);
      |
      | InferenceMode::is_enabled() is in perf critical
      | path (TensorImpl constructor) so it worths
      | a separate TLS to skip the DispatchKeySet
      | check.
      |
      */
    pub fn is_enabled(&mut self) -> bool {
        
        todo!();
        /*
            return InferenceMode_enabled;
        */
    }
    
    /**
      | _set_enabled() is not user facing and
      | should be only used in ThreadLocalState.cpp.
      |
      */
    pub fn set_enabled(&mut self, enabled: bool)  {
        
        todo!();
        /*
            InferenceMode_enabled = enabled;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/InferenceMode.cpp]

lazy_static!{
    /*
    thread_local bool InferenceMode_enabled = false;
    */
}
