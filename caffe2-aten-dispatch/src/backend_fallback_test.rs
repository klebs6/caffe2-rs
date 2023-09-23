/*!
  | This test file gives an example of a simple use
  | case for "wrapper" and "mode" style tensor type
  | ids.  In both cases, the implementation of the
  | wrapper/mode simply passes through the call to
  | underlying JIT implementation (so the
  | wrapper/mode doesn't actually do anything), but
  | this could be used as a starting point to do
  | more interesting things.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/backend_fallback_test.cpp]

/**
  | Global counter for ease of testing
  | 
  |
  */
lazy_static!{
    /*
    static i64 override_call_count = 0;
    */
}

// Mode implementation
pub fn generic_mode_fallback(
        op:    &OperatorHandle,
        stack: *mut JitStack)  {
    
    todo!();
        /*
            override_call_count++;
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);
      op.callBoxed(stack);
        */
}

/**
  | Wrapper implementation
  |
  */
pub struct GenericWrapperTensorImpl {
    base: TensorImpl,
    rep:  Tensor,
}

impl GenericWrapperTensorImpl {
    
    pub fn new(rep: Tensor) -> Self {
    
        todo!();
        /*


            : TensorImpl(
            c10::DispatchKeySet(c10::DispatchKey::TESTING_ONLY_GenericWrapper),
            rep.dtype(),
            rep.device()
            // TODO: propagate size!
          )
        , rep_(std::move(rep))
        */
    }
}

pub fn generic_wrapper_fallback(
        op:    &OperatorHandle,
        stack: *mut JitStack)  {
    
    todo!();
        /*
            override_call_count++;

      auto num_arguments = op.schema().arguments().size();
      auto num_returns = op.schema().returns().size();

      // Unwrap all arguments
      auto args = torch::jit::pop(*stack, num_arguments);
      for (usize i = 0; i < num_arguments; i++) {
        // TODO: Handle tensor list
        if (args[i].isTensor()) {
          auto* impl = args[i].unsafeToTensorImpl();
          if (impl->key_set().has(DispatchKey::TESTING_ONLY_GenericWrapper)) {
            auto* wrapper = static_cast<GenericWrapperTensorImpl*>(impl);
            torch::jit::push(*stack, wrapper->rep_);  // no move!
          } else {
            torch::jit::push(*stack, std::move(args[i]));
          }
        } else {
          torch::jit::push(*stack, std::move(args[i]));
        }
      }

      op.callBoxed(stack);

      // Rewrap outputs
      auto rets = torch::jit::pop(*stack, num_returns);
      for (usize i = 0; i < num_returns; i++) {
        // TODO: Handle tensor list
        if (rets[i].isTensor()) {
          torch::jit::push(*stack, at::detail::make_tensor<GenericWrapperTensorImpl>(std::move(rets[i]).toTensor()));  // yes move!
        } else {
          torch::jit::push(*stack, std::move(rets[i]));
        }
      }
        */
}

#[cfg(not(ATEN_CPU_STATIC_DISPATCH))]
pub mod not_aten_cpu_static_dispatch {

    use super::*;

    #[test] fn backend_fallback_test_with_mode() {
        todo!();
        /*
        
          auto m = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode);
          m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());

          c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);

          override_call_count = 0;
          Tensor a = ones({5, 5}, kDouble);
          Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
          ASSERT_EQ(override_call_count, 2);

        */
    }

    #[test] fn backend_fallback_test_with_wrapper() {
        todo!();
        /*
        
          auto m = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
          m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_wrapper_fallback>());

          override_call_count = 0;
          Tensor a = at::detail::make_tensor<GenericWrapperTensorImpl>(ones({5, 5}, kDouble));
          Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
          ASSERT_EQ(override_call_count, 1);

        */
    }

    #[test] fn backend_fallback_test_fallthrough() {
        todo!();
        /*
        
          auto m = MAKE_TORCH_LIBRARY_IMPL(aten, TESTING_ONLY_GenericMode);
          m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());

          auto gm = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode);
          gm.fallback(torch::CppFunction::makeFallthrough());

          c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);

          override_call_count = 0;
          // Doesn't trigger, as we fallthrough
          Tensor a = zeros({5, 5}, kDouble);
          ASSERT_EQ(override_call_count, 0);
          // Does trigger, because we explicitly set it
          Tensor b = mul(a, a);
          ASSERT_EQ(override_call_count, 1);

        */
    }
}
