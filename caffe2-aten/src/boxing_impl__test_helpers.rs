crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/test_helpers.h]

#[inline] pub fn make_stack(inputs: Inputs) -> Vec<IValue> {
    
    todo!();
        /*
            return {std::forward<Inputs>(inputs)...};
        */
}

#[inline] pub fn dummy_tensor<T: Into<DispatchKeySet>>(
        ks:            T,
        requires_grad: bool) -> Tensor {

    let requires_grad: bool = requires_grad.unwrap_or(false);

    todo!();
        /*
            auto* allocator = c10::GetCPUAllocator();
      i64 nelements = 1;
      auto dtype = caffe2::TypeMeta::Make<float>();
      i64 size_bytes = nelements * dtype.itemsize();
      auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
          c10::StorageImpl::use_byte_Size(),
          size_bytes,
          allocator->allocate(size_bytes),
          allocator,
          /*resizable=*/true);
      at::Tensor t = at::detail::make_tensor<c10::TensorImpl>(storage_impl, ks, dtype);
      // TODO: We add this to simulate the ideal case where we only have Autograd backend keys
      //       on Tensor when it requires grad. But currently Autograd keys are added in TensorImpl
      //       constructor by default.
      if (!requires_grad) {
        t.unsafeGetTensorImpl()->remove_autograd_key();
      }
      return t;
        */
}

#[inline] pub fn call_op<Args>(
        op:   &OperatorHandle,
        args: Args) -> Vec<IValue> {
    
    todo!();
        /*
            auto stack = makeStack(std::forward<Args>(args)...);
      op.callBoxed(&stack);
      return stack;
        */
}

#[inline] pub fn call_op_unboxed<Result,Args>(
        op:   &OperatorHandle,
        args: Args) -> Result {
    
    todo!();
        /*
            return op.typed<Result(Args...)>().call(std::forward<Args>(args)...);
        */
}

#[inline] pub fn call_op_unboxed_with_dispatch_key<Result,Args>(
        op:           &OperatorHandle,
        dispatch_key: DispatchKey,
        args:         Args) -> Result {
    
    todo!();
        /*
            return op.typed<Result(Args...)>().callWithDispatchKey(dispatchKey, std::forward<Args>(args)...);
        */
}

#[inline] pub fn call_op_unboxed_with_precomputed_dispatch_key_set<Result,Args>(
        op:   &OperatorHandle,
        ks:   DispatchKeySet,
        args: Args) -> Result {
    
    todo!();
        /*
            return op.typed<Result(Args...)>().redispatch(ks, std::forward<Args>(args)...);
        */
}

#[inline] pub fn expect_doesnt_find_kernel(
        op_name:      *const u8,
        dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
      EXPECT_ANY_THROW(
        callOp(*op, dummyTensor(dispatch_key), 5);
      );
        */
}

#[inline] pub fn expect_doesnt_find_operator(op_name: *const u8)  {
    
    todo!();
        /*
            auto op = c10::Dispatcher::singleton().findSchema({op_name, ""});
      EXPECT_FALSE(op.has_value());
        */
}

#[inline] pub fn expect_throws<Exception, Functor>(
        functor:                 Functor,
        expect_message_contains: *const u8)  {

    todo!();
        /*
            try {
        std::forward<Functor>(functor)();
      } catch (const Exception& e) {
        EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
        return;
      }
      ADD_FAILURE() << "Expected to throw exception containing \""
        << expectMessageContains << "\" but didn't throw";
        */
}

pub fn expect_list_equals<E,A>(
    expected: E,
    actual:   A) where E: IntoIterator, A: IntoIterator  {

    todo!();
        /*
      EXPECT_EQ(expected.size(), actual.size());
      for (usize i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], actual.get(i));
      }
        */
}

/**
  | NB: This is not really sound, but all
  | of the type sets constructed here are
  | singletons so it's fine
  |
  */
#[inline] pub fn extract_dispatch_key(t: &Tensor) -> DispatchKey {
    
    todo!();
        /*
            return legacyExtractDispatchKey(t.key_set());
        */
}
