crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/kernel_stackbased_test.cpp]

pub fn error_kernel(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            EXPECT_TRUE(false); // this kernel should never be called
        */
}

pub fn increment_kernel(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            int input = torch::jit::pop(*stack).toInt();
      torch::jit::pop(*stack); // pop the dummy tensor
      torch::jit::push(*stack, input + 1);
        */
}

pub fn decrement_kernel(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            int input = torch::jit::pop(*stack).toInt();
      torch::jit::pop(*stack); // pop the dummy tensor
      torch::jit::push(*stack, input - 1);
        */
}

lazy_static!{
    /*
    bool called_redispatching_kernel = false;
    */
}

pub fn redispatching_kernel_with_dispatch_key_set(
        op:    &OperatorHandle,
        ks:    DispatchKeySet,
        stack: *mut Stack)  {
    
    todo!();
        /*
            // this kernel is a no-op- it just redispatches to the lower-priority kernel
      called_redispatching_kernel = true;
      auto updated_ks = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::TESTING_ONLY_GenericWrapper);
      op.redispatchBoxed(updated_ks, stack);
        */
}

pub fn expect_calls_increment(ks: DispatchKeySet)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      auto result = callOp(*op, dummyTensor(ks), 5);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(6, result[0].toInt());
        */
}

pub fn expect_calls_increment_with_key(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            expectCallsIncrement(c10::DispatchKeySet(dispatch_key));
        */
}

pub fn expect_calls_increment_unboxed(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      i64 result = callOpUnboxed<i64, at::Tensor, i64>(*op, dummyTensor(dispatch_key), 5);
      EXPECT_EQ(6, result);
        */
}

pub fn expect_calls_decrement(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      auto result = callOp(*op, dummyTensor(dispatch_key), 5);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(4, result[0].toInt());
        */
}

#[test] fn operator_registration_test_stack_based_kernel_given_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_stack_based_kernel_given_multiple_operators_and_kernels_when_registered_in_one_registrar_then_calls_right() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                          .kernel<&errorKernel>(DispatchKey::CUDA))
          .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                          .kernel<&errorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_stack_based_kernel_given_multiple_operators_and_kernels_when_registered_in_registrars_then_calls_right() {
    todo!();
    /*
    
      auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<&errorKernel>(DispatchKey::CUDA));
      auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<&errorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_stack_based_kernel_given_when_runs_out_of_scope_then_cannot_be_called_anymore() {
    todo!();
    /*
    
      {
        auto m = MAKE_TORCH_LIBRARY(_test);
        m.def("_test::my_op(Tensor dummy, int input) -> int");
        auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
        m_cpu.impl("my_op", DispatchKey::CPU, torch::CppFunction::makeFromBoxedFunction<incrementKernel>());
        {
          auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
          m_cuda.impl("my_op", DispatchKey::CUDA, torch::CppFunction::makeFromBoxedFunction<decrementKernel>());

          // assert that schema and cpu kernel are present
          expectCallsIncrement(DispatchKey::CPU);
          expectCallsDecrement(DispatchKey::CUDA);
        }

        // now registrar2 is destructed. Assert that schema is still present but cpu kernel is not
        expectCallsIncrement(DispatchKey::CPU);
        expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
      }

      // now both registrars are destructed. Assert that the whole schema is gone
      expectDoesntFindOperator("_test::my_op");

    */
}

lazy_static!{
    /*
    bool called = false;
    */
}

pub fn kernel_without_inputs(
        _0: &OperatorHandle,
        _1: *mut Stack)  {
    
    todo!();
        /*
            called = true;
        */
}

#[test] fn operator_registration_test_stack_based_kernel_given_fallback_without_any_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<&kernelWithoutInputs>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op);
      EXPECT_TRUE(called);

    */
}

pub fn kernel_without_tensor_inputs(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            stack->back() = stack->back().toInt() + 1;
        */
}

#[test] fn operator_registration_test_stack_based_kernel_given_fallback_without_tensor_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<&kernelWithoutTensorInputs>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, 3);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(4, outputs[0].toInt());

    */
}

pub fn kernel_for_schema_inference(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
        
        */
}

#[test] fn operator_registration_test_stack_based_kernel_given_when_registered_without_specifying_schema_then_fails_because_it_cannot_infer_from() {
    todo!();
    /*
    
      expectThrows<c10::Error>([] {
          RegisterOperators().op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<&kernelForSchemaInference>());
      }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::no_schema_specified");

    */
}

#[test] fn operator_registration_test_stack_based_kernel_given_when_registered_then_can_also_be_called_unboxed() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
      expectCallsIncrementUnboxed(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_stack_based_kernel_call_kernels_with_dispatch_key_set_convention_redispatches_to_lower_priority() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("my_op(Tensor dummy, int input) -> int");
      auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
      m_cpu.fallback(torch::CppFunction::makeFromBoxedFunction<&incrementKernel>());
      auto m_testing = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
      m_testing.fallback(torch::CppFunction::makeFromBoxedFunction<&redispatchingKernel_with_DispatchKeySet>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());

      auto testing_cpu_set = c10::DispatchKeySet()
                                        .add(c10::DispatchKey::TESTING_ONLY_GenericWrapper)
                                        .add(c10::DispatchKey::CPU);
      called_redispatching_kernel = false;

      // call CPU (and not TESTING_ONLY_GenericWrapper)
      expectCallsIncrement(DispatchKey::CPU);
      ASSERT_FALSE(called_redispatching_kernel);

      // call TESTING_ONLY_GenericWrapper -> call CPU
      expectCallsIncrement(testing_cpu_set);
      ASSERT_TRUE(called_redispatching_kernel);

    */
}
