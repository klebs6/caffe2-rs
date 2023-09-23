/*!
  | This file contains some general registration
  | test cases.
  | 
  | More detailed test cases containing
  | different APIs for registering kernels
  | are found in other files in this directory.
  |
  | This file intentionally tests some deprecated
  | APIs
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/op_registration_test.cpp]

pub struct DummyKernel {
    base: OperatorKernel,
}

impl DummyKernel {
    
    pub fn invoke(&mut self, _0: Tensor)  {
        
        todo!();
        /*
        
        */
    }
}

pub struct MockKernel {
    base:   OperatorKernel,
    called: *mut bool,
}

impl MockKernel {
    
    pub fn new(called: *mut bool) -> Self {
    
        todo!();
        /*
        : called(called),

        
        */
    }
    
    pub fn invoke(&mut self, _0: Tensor)  {
        
        todo!();
        /*
            *called_ = true;
        */
    }
}

#[test] fn operator_registration_test_when_registering_with_schema_before_kernel_in_options_object_then_can_be_called() {
    todo!();
    /*
    
      bool called = false;
      auto registrar = RegisterOperators().op(RegisterOperators::options().schema("_test::dummy(Tensor dummy) -> ()").catchAllKernel<MockKernel>(&called));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_when_registering_with_schema_after_kernel_in_options_object_then_can_be_called() {
    todo!();
    /*
    
      bool called = false;
      auto registrar = RegisterOperators().op(RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy(Tensor dummy) -> ()"));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_when_registering_with_name_before_kernel_in_options_object_then_can_be_called() {
    todo!();
    /*
    
      bool called = false;
      auto registrar = RegisterOperators().op(RegisterOperators::options().schema("_test::dummy").catchAllKernel<MockKernel>(&called));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_when_registering_with_name_after_kernel_in_options_object_then_can_be_called() {
    todo!();
    /*
    
      bool called = false;
      auto registrar = RegisterOperators().op(RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy"));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_when_registering_without_schema_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([] {
        RegisterOperators().op(RegisterOperators::options().catchAllKernel<DummyKernel>());
      }, "In operator registration: Tried to register an operator without specifying a schema or operator name.");

    */
}

#[test] fn operator_registration_test_when_calling_op_with_wrong_dispatch_key_then_fails() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().kernel<DummyKernel>(DispatchKey::CPU));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::CUDA));
      }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
      " backend.");

    */
}

#[test] fn operator_registration_test_given_op_with_catchall_kernel_when_calling_then_calls() {
    todo!();
    /*
    
      bool called = false;
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().catchAllKernel<MockKernel>(&called));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernel_thenFails) {
//   bool called = false;
//   auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   expectThrows<Error>([&] {
//     RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().kernel<MockKernel>(DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<Error>([&] {
//     auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
//       .catchAllKernel<MockKernel>(&called)
//       .kernel<MockKernel>(DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

#[test] fn operator_registration_test_given_op_with_dispatched_kernel_out_of_scope_when_registering_catchall_and_calling_then_calls() {
    todo!();
    /*
    
      bool called = false;
      {
        auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().kernel<MockKernel>(DispatchKey::CPU, &called));
      }

      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().catchAllKernel<MockKernel>(&called));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called);

    */
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernel_thenFails) {
//   bool called = false;
//   auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().kernel<MockKernel>(DispatchKey::CPU, &called));
//   expectThrows<Error>([&] {
//     RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }
//
// TEST(OperatorRegistrationTest, givenOpWithDispatchedKernel_whenRegisteringCatchallKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<Error>([&] {
//     auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
//       .kernel<MockKernel>(DispatchKey::CPU, &called)
//       .catchAllKernel<MockKernel>(&called));
//   }, "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys CPU. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is _test::dummy");
// }

#[test] fn operator_registration_test_given_op_with_catchall_kernel_out_of_scope_when_registering_dispatched_and_calling_then_calls() {
    todo!();
    /*
    
      bool called = false;
      {
        auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().catchAllKernel<MockKernel>(&called));
      }

      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options().kernel<MockKernel>(DispatchKey::CPU, &called));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      EXPECT_FALSE(called);
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_given_op_without_kernels_when_registering_with_schema_then_only_registers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered
      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::CPU));
      }, "Could not run '_test::dummy' with arguments from the 'CPU'"
      " backend.");

    */
}

#[test] fn operator_registration_test_given_op_without_kernels_when_registering_schema_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([&] {
        RegisterOperators().op("_test::dummy");
      }, "Cannot infer operator schema in registration of operator _test::dummy because there is no kernel specified.");

    */
}

#[test] fn operator_registration_test_given_op_without_kernels_when_running_out_of_scope_then_schema_is_gone() {
    todo!();
    /*
    
      {
        auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
      }

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      EXPECT_FALSE(op.has_value());

    */
}

#[test] fn operator_registration_test_given_op_without_kernels_tensor_inputs_when_registering_then_registers() {
    todo!();
    /*
    
      // as long as we don't register non-catchall kernels, ops without tensor arguments are fine
      auto registrar = RegisterOperators().op("_test::dummy() -> ()");

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

    */
}

#[test] fn operator_registration_test_given_multiple_kernels_with_same_dispatch_key_when_registering_in_op_call_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([&] {
        auto registrar = RegisterOperators()
            .op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
                .kernel<DummyKernel>(DispatchKey::CPU)
                .kernel<DummyKernel>(DispatchKey::CPU));
      }, "In operator registration: Tried to register multiple kernels with same dispatch key CPU for operator schema _test::dummy");

    */
}

#[test] fn operator_registration_test_given_multiple_catchall_kernels_when_registering_in_same_op_call_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([&] {
        auto registrar = RegisterOperators()
            .op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
                .catchAllKernel<DummyKernel>()
                .catchAllKernel<DummyKernel>());
      }, "Tried to register multiple catch-all kernels for operator schema _test::dummy");

    */
}

#[test] fn operator_registration_test_when_registering_cpu_tensor_type_then_can_only_call_unboxed_with_dispatch_key() {
    todo!();
    /*
    
      bool called_kernel_cpu = false;
      auto registrar= RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<MockKernel>(DispatchKey::CPU, &called_kernel_cpu));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

      // Ensure that dispatcher doesn't take the dispatch key from the tensor but from the direct argument instead.
      called_kernel_cpu = false;
      callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, DispatchKeySet(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called_kernel_cpu);

      // Ensure that disptach key from tensor is not used here.
      called_kernel_cpu = false;
      expectThrows<Error>([&] {
        callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, DispatchKeySet(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
      }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
      " backend.");

    */
}

#[test] fn operator_registration_test_when_registering_multiple_kernels_in_same_op_call_and_calling_then_calls_correct_kernel() {
    todo!();
    /*
    
      bool called_kernel1 = false;
      bool called_kernel2 = false;
      auto registrar0 = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<MockKernel>(DispatchKey::CPU, &called_kernel1)
        .kernel<MockKernel>(DispatchKey::CUDA, &called_kernel2));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

      called_kernel1 = called_kernel2 = false;
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_kernel1);
      EXPECT_FALSE(called_kernel2);

      called_kernel1 = called_kernel2 = false;
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_FALSE(called_kernel1);
      EXPECT_TRUE(called_kernel2);

      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::XLA));
      }, "Could not run '_test::dummy' with arguments from the 'XLA'"
      " backend.");

      // also assert that the error message contains the available tensor type ids, but don't assert their order
      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::XLA));
      }, "CPU");
      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::XLA));
      }, "CUDA");

    */
}

lazy_static!{
    /*
    bool called_stackbased_kernel = false;
    */
}

pub fn stack_based_kernel(
        _0:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            called_stackbased_kernel = true;
        */
}

#[test] fn operator_registration_test_when_registering_multiple_kernels_by_name_and_none_can_infer_schema_then_fails() {
    todo!();
    /*
    
      bool called_kernel = false;
      expectThrows<Error>([&] {
        auto registrar1 = RegisterOperators().op("_test::dummy", RegisterOperators::options()
          .kernel<&stackBasedKernel>(DispatchKey::CPU)
          .kernel<&stackBasedKernel>(DispatchKey::CUDA)
          .kernel<&stackBasedKernel>(DispatchKey::XLA));
      }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::dummy");

    */
}

#[test] fn operator_registration_test_when_registering_multiple_kernels_by_schema_and_none_can_infer_then_succeeds() {
    todo!();
    /*
    
      bool called_kernel = false;
      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<&stackBasedKernel>(DispatchKey::CPU)
        .kernel<&stackBasedKernel>(DispatchKey::CUDA)
        .kernel<&stackBasedKernel>(DispatchKey::XLA));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::XLA));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

    */
}

#[test] fn operator_registration_test_when_registering_multiple_kernels_by_name_and_only_one_can_infer_schema_then_succeeds() {
    todo!();
    /*
    
      bool called_kernel = false;
      auto registrar1 = RegisterOperators().op("_test::dummy", RegisterOperators::options()
        .kernel<&stackBasedKernel>(DispatchKey::CPU)
        .kernel<MockKernel>(DispatchKey::CUDA, &called_kernel)
        .kernel<&stackBasedKernel>(DispatchKey::XLA));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_FALSE(called_stackbased_kernel);
      EXPECT_TRUE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::XLA));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

    */
}

#[test] fn operator_registration_test_when_registering_multiple_kernels_by_schema_and_only_one_can_infer_then_succeeds() {
    todo!();
    /*
    
      bool called_kernel = false;
      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<&stackBasedKernel>(DispatchKey::CPU)
        .kernel<MockKernel>(DispatchKey::CUDA, &called_kernel)
        .kernel<&stackBasedKernel>(DispatchKey::XLA));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value()); // assert schema is registered

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_FALSE(called_stackbased_kernel);
      EXPECT_TRUE(called_kernel);

      called_kernel = called_stackbased_kernel = false;
      callOp(*op, dummyTensor(DispatchKey::XLA));
      EXPECT_TRUE(called_stackbased_kernel);
      EXPECT_FALSE(called_kernel);

    */
}

pub struct DummyKernelWithIntParam {
    base: OperatorKernel,
}

impl DummyKernelWithIntParam {
    
    pub fn invoke(&mut self, 
        _0: Tensor,
        _1: i64)  {
        
        todo!();
        /*
        
        */
    }
}

#[test] fn operator_registration_test_when_registering_mismatching_kernels_in_same_op_call_then_fails() {
    todo!();
    /*
    
      bool called_kernel = false;
      expectThrows<Error>([&] {
        auto registrar1 = RegisterOperators().op("_test::dummy", RegisterOperators::options()
          .kernel<DummyKernelWithIntParam>(DispatchKey::CPU)
          .kernel<MockKernel>(DispatchKey::CUDA, &called_kernel));
      }, "Mismatch in kernel C++ signatures");

    */
}

pub fn backend_fallback_kernel(
        op:    &OperatorHandle,
        stack: *mut Stack)  {
    
    todo!();
        /*
            (*stack)[1] = (*stack)[1].toString()->string() + op.schema().name();
        */
}

#[test] fn operator_registration_test_when_registering_backend_fallback_kernel_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = Dispatcher::singleton().registerFallback(DispatchKey::CPU, KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());

    */
}

#[test] fn operator_registration_test_when_registering_backend_fallback_kernel_for_wrong_then_cannot_be_called() {
    todo!();
    /*
    
      auto registrar = Dispatcher::singleton().registerFallback(DispatchKey::CUDA, KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      expectThrows<Error>([&] {
        auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      }, "Could not run '_test::dummy' with arguments from the 'CPU' backend.");

    */
}

lazy_static!{
    /*
    bool called = false;
    */
}

#[test] fn operator_registration_test_when_registering_backend_fallback_kernel_and_regular_for_different_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = Dispatcher::singleton().registerFallback(DispatchKey::CPU, KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", RegisterOperators::options()
          .kernel(DispatchKey::CUDA, [] (Tensor, string) {
            called = true;
          }));
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto stack = callOp(*op, dummyTensor(DispatchKey::CUDA), "hello ");
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_when_registering_backend_fallback_kernel_and_regular_for_different_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = Dispatcher::singleton().registerFallback(DispatchKey::CPU, KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", RegisterOperators::options()
          .kernel(DispatchKey::CUDA, [] (Tensor, string) {
            called = true;
          }));
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      EXPECT_FALSE(called);
      EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());

    */
}

#[test] fn operator_registration_test_when_registering_backend_fallback_kernel_and_regular_for_same_then_calls() {
    todo!();
    /*
    
      auto registrar = Dispatcher::singleton().registerFallback(DispatchKey::CPU, KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (Tensor, string) {
            called = true;
          }));
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      EXPECT_TRUE(called);

    */
}

lazy_static!{
    /*
    bool called_autograd = false;
    bool called_nonautograd = false;
    */
}

pub fn nonautograd_kernel(a: Tensor)  {
    
    todo!();
        /*
            called_nonautograd = true;
        */
}

pub fn autograd_kernel(a: Tensor)  {
    
    todo!();
        /*
            called_autograd = true;
        */
}

#[test] fn operator_registration_test_when_registering_autograd_kernel_then_can_call() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called_autograd = false;
      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::CPU));
      }, "Could not run '_test::dummy' with arguments from the 'CPU'"
      " backend.");

      op->typed<void(Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_TRUE(called_autograd);

    */
}

#[test] fn operator_registration_test_when_registering_autograd_kernel_with_regular_then_can_call() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<decltype(nonautograd_kernel), nonautograd_kernel>(DispatchKey::CPU)
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_FALSE(called_nonautograd);
      EXPECT_TRUE(called_autograd);

    */
}

#[test] fn operator_registration_test_when_registering_autograd_kernel_with_catch_all_then_can_call() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      // catchAll now maps to CompositeImplicitAutograd which has higher precedence than Autograd
      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);

    */
}

#[test] fn operator_registration_test_when_registering_autograd_kernel_with_catch_all_then_can_call_catchall() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);

    */
}

#[test] fn operator_registration_test_autograd_backend_overrides_kernel() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradCPU)
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::CPU));
      }, "Could not run '_test::dummy' with arguments from the 'CPU'"
      " backend.");

      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::CUDA));
      }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
      " backend.");

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
      EXPECT_TRUE(called_autograd);
      EXPECT_FALSE(called_nonautograd);

    */
}

#[test] fn operator_registration_test_autograd_xla_overrides_kernel() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
        .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradXLA)
        .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());

      expectThrows<Error>([&] {
        callOp(*op, dummyTensor(DispatchKey::XLA));
      }, "Could not run '_test::dummy' with arguments from the 'XLA'"
      " backend.");

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);

      called_nonautograd = called_autograd = false;
      op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_TRUE(called_autograd);
      EXPECT_FALSE(called_nonautograd);

    */
}

#[test] fn operator_registration_test_when_register_with_xla_kernel_and_catch_all_autograd_is_not_filled() {
    todo!();
    /*
    
      {
        auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
          .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

        auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
        ASSERT_TRUE(op.has_value());

        called_nonautograd = called_autograd = false;
        op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        EXPECT_TRUE(called_nonautograd);
        EXPECT_FALSE(called_autograd);

        called_nonautograd = called_autograd = false;
        op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA));
        EXPECT_FALSE(called_autograd);
        EXPECT_TRUE(called_nonautograd);
      }
      {
        auto registrar = RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", RegisterOperators::options()
          .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::XLA)
          .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

        auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
        ASSERT_TRUE(op.has_value());

        // When there's direct registration to XLA backend, AutogradXLA doesn't pick up catchAll
        // kernel in precompute but just keep fallthrough kernel from backend fallback.
        // Thus it falls through AutogradXLA and reaches the kernel at XLA key.
        called_nonautograd = called_autograd = false;
        op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        EXPECT_FALSE(called_nonautograd);
        EXPECT_TRUE(called_autograd);

        called_nonautograd = called_autograd = false;
        op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::XLA));
        EXPECT_TRUE(called_autograd);
        EXPECT_FALSE(called_nonautograd);
      }

    */
}

#[test] fn operator_registration_test_given_lambda_kernel_when_registering_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([] {
        auto registrar = RegisterOperators().op("_test::dummy", RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (const i64&) {})
          .kernel(DispatchKey::CUDA, [] (i64) {}));
      }, "Mismatch in kernel C++ signatures");

    */
}

#[test] fn operator_registration_test_given_lambda_kernel_when_registering_catch_all_and_backend_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([] {
        auto registrar = RegisterOperators().op("_test::dummy", RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (const i64&) {})
          .catchAllKernel([] (i64) {}));
      }, "Mismatch in kernel C++ signatures");

    */
}

#[test] fn operator_registration_test_given_lambda_kernel_when_registering_backend_and_catch_all_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      expectThrows<Error>([] {
        auto registrar = RegisterOperators().op("_test::dummy", RegisterOperators::options()
          .catchAllKernel([] (const i64&) {})
          .kernel(DispatchKey::CPU, [] (i64) {}));
      }, "Mismatch in kernel C++ signatures");

    */
}

#[test] fn operator_registration_test_given_lambda_kernel_when_accessing_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy", RegisterOperators::options()
        .kernel(DispatchKey::CPU, [] (i64) {}));
      expectThrows<Error>([] {
        Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
          .typed<void(const i64&)>();
      }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");

    */
}

#[test] fn operator_registration_test_given_lambda_kernel_when_accessing_catch_all_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::dummy", RegisterOperators::options()
        .catchAllKernel([] (i64) {}));
      expectThrows<Error>([] {
        Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
          .typed<void(const i64&)>();
      }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");

    */
}

#[test] fn operator_registration_test_given_torch_library_when_registering_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("dummy(int a) -> ()");
      m.impl("dummy", DispatchKey::CPU, [] (i64) {});
      expectThrows<Error>([&] {
        m.impl("dummy", DispatchKey::CUDA, [] (const i64&) {});
      }, "Mismatch in kernel C++ signatures");

    */
}

#[test] fn operator_registration_test_given_torch_library_when_accessing_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("dummy(int a) -> ()");
      m.impl("dummy", DispatchKey::CPU, [] (i64) {});
      expectThrows<Error>([] {
        Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
          .typed<void(const i64&)>();
      }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");

    */
}

#[test] fn operator_registration_test_given_torch_library_when_accessing_catch_all_with_mismatching_cpp_signatures_then_fails() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("dummy(int a) -> ()", [] (i64) {});
      expectThrows<Error>([] {
        Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
          .typed<void(const i64&)>();
      }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");

    */
}

/**
 | This is used to check that a given type works
 | correctly when passed as input to or as output
 | from a kernel.
 |
 | Call ArgTypeTestKernel<Input,
 | Output>::test(input, inputExpectation, output,
 | outputExpectation, schema) to test that
 | a kernel with `Input` as input type and
 | `Output` as output types, when called with
 | `input` fulfills `inputExpectation` inside the
 | kernel, then returns `output` and the returned
 | value fulfills `outputExpectation`.
 |
 | `inputExpectation` and `outputExpectation`
 | should be lambdas that run googletest expect
 | macros (or use other ways to assert the
 | expectation is met).
 |
 | Optionally, you can specify the argument list
 | part of a function schema (e.g. "(Tensor a) ->
 | Tensor") as an additional argument to use when
 | registering the kernel. In this case, the
 | operator registration logic will check that the
 | kernel function signature matches the one you
 | specified.
 */
struct TestModernAPI {}
struct TestLegacyAPI {}
struct TestModernAndLegacyAPI {}

pub struct ArgTypeTestKernel<InputType,OutputType = InputType> {
    base:              OperatorKernel,
    input:             InputType,
    input_expectation: fn(_0: &InputType) -> (),
    output:            OutputType,
    schema:            String,
}

impl ArgTypeTestKernel<InputType,OutputType> {
    
    pub fn new(
        input:             InputType,
        input_expectation: fn(_0: &InputType) -> (),
        output:            OutputType) -> Self {
    
        todo!();
        /*


            : input_(move(input)), inputExpectation_(move(inputExpectation)), output_(move(output))
        */
    }
    
    pub fn invoke(&self, input: InputType) -> OutputType {
        
        todo!();
        /*
            inputExpectation_(move(input));
        return output_;
        */
    }
    
    pub fn test(
        _0:                 TestModernAndLegacyAPI,
        input:              InputType,
        input_expectation:  fn(_0: &InputType) -> (),
        output:             OutputType,
        output_expectation: fn(_0: &Stack) -> (),
        schema:             &String)  {
        
        todo!();
        /*
            test(TestModernAPI(), input, inputExpectation, output, outputExpectation, schema);
        test(TestLegacyAPI(), input, inputExpectation, output, outputExpectation, schema);
        */
    }
    
    pub fn test(
        _0:                 TestModernAPI,
        input:              InputType,
        input_expectation:  fn(_0: &InputType) -> (),
        output:             OutputType,
        output_expectation: fn(_0: &Stack) -> (),
        schema:             &String)  {
        
        todo!();
        /*
            return test_([&] {
          return RegisterOperators().op("_test::my_op" + schema, RegisterOperators::options().catchAllKernel<ArgTypeTestKernel>(input, inputExpectation, output));
        }, input, inputExpectation, output, outputExpectation, schema);
        */
    }
    
    pub fn test(
        _0:                 TestLegacyAPI,
        input:              InputType,
        input_expectation:  fn(_0: &InputType) -> (),
        output:             OutputType,
        output_expectation: fn(_0: &Stack) -> (),
        schema:             &String)  {
        
        todo!();
        /*
            return test_([&] {
          return RegisterOperators().op("_test::my_op" + schema, [=] (InputType input) -> OutputType {
            inputExpectation(move(input));
            return output;
          });
        }, input, inputExpectation, output, outputExpectation, schema);
        */
    }
    
    pub fn test(
        registration:       fn() -> RegisterOperators,
        input:              InputType,
        input_expectation:  fn(_0: &InputType) -> (),
        output:             OutputType,
        output_expectation: fn(_0: &Stack) -> (),
        schema:             &String)  {
        
        todo!();
        /*
            auto registry = registration();
        auto op = Dispatcher::singleton().findSchema({"_test::my_op", ""});
        ASSERT_TRUE(op.has_value()); // assert schema is registered
        auto actualOutput = callOp(*op, input);
        outputExpectation(actualOutput);
        */
    }
}

pub struct TestArgTypes<InputType,OutputType = InputType> {

}

impl TestArgTypes<InputType,OutputType> {
    
    pub fn test<APIType = TestModernAndLegacyAPI>(
        input:              InputType,
        input_expectation:  fn(_0: &InputType) -> (),
        output:             OutputType,
        output_expectation: fn(_0: &IValue) -> (),
        schema:             &String)  {
    
        todo!();
        /*
            // Test with explicitly specified schema
        ArgTypeTestKernel<InputType, OutputType>::test(
          APIType(), input, inputExpectation, output, [&] (const Stack& output) {
            EXPECT_EQ(1, output.size());
            outputExpectation(output[0]);
          }, schema
        );

        // Test with inferred schema
        ArgTypeTestKernel<InputType, OutputType>::test(
          APIType(), input, inputExpectation, output, [&] (const Stack& output) {
            EXPECT_EQ(1, output.size());
            outputExpectation(output[0]);
          }, ""
        );

        // Test taking argument and returning nothing
        ArgTypeTestKernel<InputType, tuple<>>::test(
          APIType(), input, inputExpectation, {}, [] (const Stack&) {}, ""
        );

        // Test taking argument and returning multiple outputs
        ArgTypeTestKernel<InputType, tuple<i64, OutputType>>::test(
          APIType(), input, inputExpectation, tuple<i64, OutputType>{3, output}, [&] (const Stack& output) {
            EXPECT_EQ(2, output.size());
            EXPECT_EQ(3, output[0].toInt());
            outputExpectation(output[1]);
          }, ""
        );
        */
    }
}

#[test] fn operator_registration_test_available_arg_types() {
    todo!();
    /*
    
      // TODO Test Scalar

      // primitive types
      testArgTypes<double>::test(
        1.5, [] (const double& v) {EXPECT_EQ(1.5, v);},
        2.5, [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
        "(float a) -> float");
      testArgTypes<i64>::test(
        1, [] (const i64& v) {EXPECT_EQ(1, v);},
        2, [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
        "(int a) -> int");
      testArgTypes<bool>::test(
        true, [] (const bool& v) {EXPECT_EQ(true, v);},
        false, [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
        "(bool a) -> bool");
      testArgTypes<bool>::test(
        false, [] (const bool& v) {EXPECT_EQ(false, v);},
        true, [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
        "(bool a) -> bool");
      testArgTypes<string>::test(
        "string1", [] (const string& v) {EXPECT_EQ("string1", v);},
        "string2", [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
        "(str a) -> str");
      testArgTypes<Tensor>::test(
        dummyTensor(DispatchKey::CPU), [] (const Tensor& v) {EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v));},
        dummyTensor(DispatchKey::CUDA), [] (const IValue& v) {EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
        "(Tensor a) -> Tensor");

      // optional types (with has_value() == true)
      testArgTypes<optional<double>>::test(
        optional<double>(1.5), [] (const optional<double>& v) {EXPECT_EQ(1.5, v.value());},
        optional<double>(2.5), [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
        "(float? a) -> float?");
      testArgTypes<optional<i64>>::test(
        optional<i64>(1), [] (const optional<i64>& v) {EXPECT_EQ(1, v.value());},
        optional<i64>(2), [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
        "(int? a) -> int?");
      testArgTypes<optional<bool>>::test(
        optional<bool>(true), [] (const optional<bool>& v) {EXPECT_EQ(true, v.value());},
        optional<bool>(false), [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
        "(bool? a) -> bool?");
      testArgTypes<optional<bool>>::test(
        optional<bool>(false), [] (const optional<bool>& v) {EXPECT_EQ(false, v.value());},
        optional<bool>(true), [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
        "(bool? a) -> bool?");
      testArgTypes<optional<string>>::test(
        optional<string>("string1"), [] (const optional<string>& v) {EXPECT_EQ("string1", v.value());},
        optional<string>("string2"), [] (const IValue& v) {EXPECT_EQ("string2", v.toString()->string());},
        "(str? a) -> str?");
      testArgTypes<optional<Tensor>>::test(
        optional<Tensor>(dummyTensor(DispatchKey::CPU)), [] (const optional<Tensor>& v) {EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.value()));},
        optional<Tensor>(dummyTensor(DispatchKey::CUDA)), [] (const IValue& v) {EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
        "(Tensor? a) -> Tensor?");

      // optional types (with has_value() == false)
      testArgTypes<optional<double>>::test(
        optional<double>(nullopt), [] (const optional<double>& v) {EXPECT_FALSE(v.has_value());},
        optional<double>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(float? a) -> float?");
      testArgTypes<optional<i64>>::test(
        optional<i64>(nullopt), [] (const optional<i64>& v) {EXPECT_FALSE(v.has_value());},
        optional<i64>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(int? a) -> int?");
      testArgTypes<optional<bool>>::test(
        optional<bool>(nullopt), [] (const optional<bool>& v) {EXPECT_FALSE(v.has_value());},
        optional<bool>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(bool? a) -> bool?");
      testArgTypes<optional<bool>>::test(
        optional<bool>(nullopt), [] (const optional<bool>& v) {EXPECT_FALSE(v.has_value());},
        optional<bool>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(bool? a) -> bool?");
      testArgTypes<optional<string>>::test(
        optional<string>(nullopt), [] (const optional<string>& v) {EXPECT_FALSE(v.has_value());},
        optional<string>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(str? a) -> str?");
      testArgTypes<optional<Tensor>>::test(
        optional<Tensor>(nullopt), [] (const optional<Tensor>& v) {EXPECT_FALSE(v.has_value());},
        optional<Tensor>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(Tensor? a) -> Tensor?");

      // list types (with empty list)
      testArgTypes<List<double>>::test(
        List<double>(), [] (const List<double>& v) {EXPECT_EQ(0, v.size());},
        List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<double>>().size());},
        "(float[] a) -> float[]");
      testArgTypes<List<i64>>::test(
        List<i64>(), [] (const List<i64>& v) {EXPECT_EQ(0, v.size());},
        List<i64>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<i64>>().size());},
        "(int[] a) -> int[]");
      testArgTypes<List<bool>>::test(
        List<bool>(), [] (const List<bool>& v) {EXPECT_EQ(0, v.size());},
        List<bool>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<bool>>().size());},
        "(bool[] a) -> bool[]");
      testArgTypes<List<string>>::test(
        List<string>(), [] (const List<string>& v) {EXPECT_EQ(0, v.size());},
        List<string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
        "(str[] a) -> str[]");

      // list types (with non-empty list)
      testArgTypes<List<double>>::test(
        List<double>({1.5, 2.5}), [] (const List<double>& v) {expectListEquals({1.5, 2.5}, v);},
        List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<List<double>>());},
        "(float[] a) -> float[]");
      testArgTypes<List<i64>>::test(
        List<i64>({1, 2}), [] (const List<i64>& v) {expectListEquals({1, 2}, v);},
        List<i64>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<List<i64>>());},
        "(int[] a) -> int[]");
      testArgTypes<List<bool>>::test(
        List<bool>({true, false}), [] (const List<bool>& v) {expectListEquals({true, false}, v);},
        List<bool>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<List<bool>>());},
        "(bool[] a) -> bool[]");
      testArgTypes<List<string>>::test(
        List<string>({"first", "second"}), [] (const List<string>& v) {expectListEquals({"first", "second"}, v);},
        List<string>({"first", "second"}), [] (const IValue& v) {
          EXPECT_EQ(2, v.toListRef().size());
          EXPECT_EQ("first", v.toListRef()[0].toStringRef());
          EXPECT_EQ("second", v.toListRef()[1].toStringRef());
        },
        "(str[] a) -> str[]");
      testArgTypes<List<Tensor>>::test(
        List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}), [] (const List<Tensor>& v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.get(0)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.get(1)));
        },
        List<Tensor>({dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU)}), [] (const IValue& v) {
          EXPECT_EQ(2, v.to<List<Tensor>>().size());
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.to<List<Tensor>>().get(0)));
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.to<List<Tensor>>().get(1)));
        },
        "(Tensor[] a) -> Tensor[]");

      // ArrayRef list types (with empty list)
      testArgTypes<ArrayRef<double>, List<double>>::test(
        ArrayRef<double>(), [] (ArrayRef<double> v) {EXPECT_EQ(0, v.size());},
        List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<double>>().size());},
        "(float[] a) -> float[]");
      testArgTypes<ArrayRef<i64>, List<i64>>::test(
        ArrayRef<i64>(), [] (ArrayRef<i64> v) {EXPECT_EQ(0, v.size());},
        List<i64>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<i64>>().size());},
        "(int[] a) -> int[]");
      testArgTypes<ArrayRef<string>, List<string>>::test(
        ArrayRef<string>(), [] (ArrayRef<string> v) {EXPECT_EQ(0, v.size());},
        List<string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
        "(str[] a) -> str[]");

      // list types (with non-empty list)
      testArgTypes<ArrayRef<double>, List<double>>::test(
        ArrayRef<double>({1.5, 2.5}), [] (ArrayRef<double> v) {expectListEquals({1.5, 2.5}, v);},
        List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<List<double>>());},
        "(float[] a) -> float[]");
      testArgTypes<ArrayRef<i64>, List<i64>>::test(
        ArrayRef<i64>({1, 2}), [] (ArrayRef<i64> v) {expectListEquals({1, 2}, v);},
        List<i64>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<List<i64>>());},
        "(int[] a) -> int[]");
      testArgTypes<ArrayRef<string>, List<string>>::test(
        ArrayRef<string>({"first", "second"}), [] (ArrayRef<string> v) {expectListEquals({"first", "second"}, v);},
        List<string>({"first", "second"}), [] (const IValue& v) {
          EXPECT_EQ(2, v.toListRef().size());
          EXPECT_EQ("first", v.toListRef()[0].toStringRef());
          EXPECT_EQ("second", v.toListRef()[1].toStringRef());
        },
        "(str[] a) -> str[]");
      testArgTypes<ArrayRef<Tensor>, List<Tensor>>::test(
        ArrayRef<Tensor>({dummyTensor(DispatchKey::CPUTensorId), dummyTensor(DispatchKey::CUDATensorId)}), [] (ArrayRef<Tensor> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
          EXPECT_EQ(DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
        },
        List<Tensor>({dummyTensor(DispatchKey::CUDATensorId), dummyTensor(DispatchKey::CPUTensorId)}), [] (const IValue& v) {
          EXPECT_EQ(2, v.to<List<Tensor>>().size());
          EXPECT_EQ(DispatchKey::CUDATensorId, extractDispatchKey(v.to<List<Tensor>>().get(0)));
          EXPECT_EQ(DispatchKey::CPUTensorId, extractDispatchKey(v.to<List<Tensor>>().get(1)));
        },
        "(Tensor[] a) -> Tensor[]");

      // array list types (with empty list)
      testArgTypes<array<double, 0>>::test(
        array<double, 0>(), [] (array<double, 0> v) {},
        array<double, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<List<double>>().size()));},
        "(float[0] a) -> float[0]");
      testArgTypes<array<i64, 0>>::test(
        array<i64, 0>(), [] (array<i64, 0> v) {},
        array<i64, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<List<i64>>().size()));},
        "(int[0] a) -> int[0]");
      testArgTypes<array<bool, 0>>::test(
        array<bool, 0>(), [] (array<bool, 0> v) {},
        array<bool, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<array<bool, 0>>().size()));},
        "(bool[0] a) -> bool[0]");
      testArgTypes<array<string, 0>>::test(
        array<string, 0>(), [] (array<string, 0> v) {EXPECT_EQ(0, v.size());},
        array<string, 0>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
        "(str[0] a) -> str[0]");

      // array list types (with non-empty list)
      testArgTypes<array<double, 2>>::test(
        array<double, 2>({1.5, 2.5}), [] (array<double, 2> v) {expectListEquals({1.5, 2.5}, v);},
        array<double, 2>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<array<double, 2>>());},
        "(float[2] a) -> float[2]");
      testArgTypes<array<i64, 2>>::test(
        array<i64, 2>({1, 2}), [] (array<i64, 2> v) {expectListEquals({1, 2}, v);},
        array<i64, 2>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<array<i64, 2>>());},
        "(int[2] a) -> int[2]");
      testArgTypes<array<bool, 2>>::test(
        array<bool, 2>({true, false}), [] (array<bool, 2> v) {expectListEquals({true, false}, v);},
        array<bool, 2>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<array<bool, 2>>());},
        "(bool[2] a) -> bool[2]");
      testArgTypes<array<string, 2>>::test(
        array<string, 2>({"first", "second"}), [] (array<string, 2> v) {expectListEquals({"first", "second"}, v);},
        array<string, 2>({"first", "second"}), [] (const IValue& v) {
          EXPECT_EQ(2, v.toListRef().size());
          EXPECT_EQ("first", v.toListRef()[0].toStringRef());
          EXPECT_EQ("second", v.toListRef()[1].toStringRef());
        },
        "(str[2] a) -> str[2]");
      testArgTypes<array<Tensor, 2>>::test(
        array<Tensor, 2>({dummyTensor(DispatchKey::CPUTensorId), dummyTensor(DispatchKey::CUDATensorId)}), [] (array<Tensor, 2> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
          EXPECT_EQ(DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
        },
        array<Tensor, 2>({dummyTensor(DispatchKey::CUDATensorId), dummyTensor(DispatchKey::CPUTensorId)}), [] (const IValue& v) {
          EXPECT_EQ(2, v.to<List<Tensor>>().size());
          EXPECT_EQ(DispatchKey::CUDATensorId, extractDispatchKey(v.to<List<Tensor>>().get(0)));
          EXPECT_EQ(DispatchKey::CPUTensorId, extractDispatchKey(v.to<List<Tensor>>().get(1)));
        },
        "(Tensor[2] a) -> Tensor[2]");

      // deprecated list types (with empty list)
      testArgTypes<vector<double>>::test<TestLegacyAPI>(
        vector<double>(), [] (const vector<double>& v) {EXPECT_EQ(0, v.size());},
        vector<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<double>>().size());},
        "(float[] a) -> float[]");
      testArgTypes<vector<i64>>::test<TestLegacyAPI>(
        vector<i64>(), [] (const vector<i64>& v) {EXPECT_EQ(0, v.size());},
        vector<i64>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<i64>>().size());},
        "(int[] a) -> int[]");
      //Note: vector<bool> is not supported, use List<bool> instead.
      testArgTypes<vector<string>>::test<TestLegacyAPI>(
        vector<string>(), [] (const vector<string>& v) {EXPECT_EQ(0, v.size());},
        vector<string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
        "(str[] a) -> str[]");

      // deprecated list types (with non-empty list)
      testArgTypes<vector<double>>::test<TestLegacyAPI>(
        vector<double>({1.5, 2.5}), [] (const vector<double>& v) {expectListEquals({1.5, 2.5}, v);},
        vector<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<List<double>>());},
        "(float[] a) -> float[]");
      testArgTypes<vector<i64>>::test<TestLegacyAPI>(
        vector<i64>({1, 2}), [] (const vector<i64>& v) {expectListEquals({1, 2}, v);},
        vector<i64>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<List<i64>>());},
        "(int[] a) -> int[]");
      //Note: vector<bool> is not supported, use List<bool> instead.
      testArgTypes<vector<string>>::test<TestLegacyAPI>(
        vector<string>({"first", "second"}), [] (const vector<string>& v) {expectListEquals({"first", "second"}, v);},
        vector<string>({"first", "second"}), [] (const IValue& v) {
          EXPECT_EQ(2, v.toListRef().size());
          EXPECT_EQ("first", v.toListRef()[0].toStringRef());
          EXPECT_EQ("second", v.toListRef()[1].toStringRef());
        },
        "(str[] a) -> str[]");
      testArgTypes<vector<Tensor>>::test<TestLegacyAPI>(
        vector<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}), [] (const vector<Tensor>& v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.at(0)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.at(1)));
        },
        vector<Tensor>({dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU)}), [] (const IValue& v) {
          EXPECT_EQ(2, v.to<List<Tensor>>().size());
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.to<List<Tensor>>().get(0)));
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.to<List<Tensor>>().get(1)));
        },
        "(Tensor[] a) -> Tensor[]");

      // Test optional of list (with nullopt)
      testArgTypes<optional<List<i64>>>::test(
        optional<List<i64>>(nullopt), [] (const optional<List<i64>>& v) {EXPECT_FALSE(v.has_value());},
        optional<List<i64>>(nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
        "(int[]? a) -> int[]?");

      // Test optional of list (with empty list)
      testArgTypes<optional<List<i64>>>::test(
        optional<List<i64>>(List<i64>({})), [] (const optional<List<i64>>& v) {EXPECT_EQ(0, v.value().size());},
        optional<List<i64>>(List<i64>({})), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<i64>>().size());},
        "(int[]? a) -> int[]?");

      // Test optional of list (with values)
      testArgTypes<optional<List<i64>>>::test(
        optional<List<i64>>(List<i64>({1, 2})), [] (const optional<List<i64>>& v) {expectListEquals({1, 2}, v.value());},
        optional<List<i64>>(List<i64>({3, 4})), [] (const IValue& v) {expectListEquals({3, 4}, v.to<List<i64>>());},
        "(int[]? a) -> int[]?");

      // Test list of optional (with empty list)
      testArgTypes<List<optional<i64>>>::test(
        List<optional<i64>>(List<optional<i64>>({})), [] (const List<optional<i64>>& v) {EXPECT_EQ(0, v.size());},
        List<optional<i64>>(List<optional<i64>>({})), [] (const IValue& v) {EXPECT_EQ(0, v.to<List<optional<i64>>>().size());},
        "(int?[] a) -> int?[]");

      // Test list of optional (with values)
      testArgTypes<List<optional<i64>>>::test(
        List<optional<i64>>(List<optional<i64>>({3, nullopt, 2})), [] (const List<optional<i64>>& v) {expectListEquals<optional<i64>>({3, nullopt, 2}, v);},
        List<optional<i64>>(List<optional<i64>>({3, nullopt, 2})), [] (const IValue& v) {expectListEquals<optional<i64>>({3, nullopt, 2}, v.to<List<optional<i64>>>());},
        "(int?[] a) -> int?[]");

      // dict types
      Dict<string, string> str_dict;
      str_dict.insert("key1", "value1");
      str_dict.insert("key2", "value2");
      testArgTypes<Dict<string, string>>::test(
        str_dict, [] (Dict<string, string> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ("value1", v.at("key1"));
          EXPECT_EQ("value2", v.at("key2"));
        },
        str_dict, [] (const IValue& v) {
          Dict<string, string> dict = toTypedDict<string, string>(v.toGenericDict());
          EXPECT_EQ(2, dict.size());
          EXPECT_EQ("value1", dict.at("key1"));
          EXPECT_EQ("value2", dict.at("key2"));
        },
        "(Dict(str, str) a) -> Dict(str, str)");
      Dict<i64, Tensor> tensor_dict;
      tensor_dict.insert(1, dummyTensor(DispatchKey::CPU));
      tensor_dict.insert(2, dummyTensor(DispatchKey::CUDA));
      testArgTypes<Dict<i64, Tensor>>::test(
        tensor_dict, [] (Dict<i64, Tensor> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.at(1)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.at(2)));
        },
        tensor_dict, [] (const IValue& v) {
          Dict<i64, Tensor> dict = toTypedDict<i64, Tensor>(v.toGenericDict());
          EXPECT_EQ(2, dict.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(dict.at(1)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
        },
        "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

      // deprecated dict types
      unordered_map<string, string> str_map;
      str_map.emplace("key1", "value1");
      str_map.emplace("key2", "value2");
      testArgTypes<unordered_map<string, string>>::test<TestLegacyAPI>(
        str_map, [] (unordered_map<string, string> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ("value1", v.at("key1"));
          EXPECT_EQ("value2", v.at("key2"));
        },
        str_map, [] (const IValue& v) {
          Dict<string, string> dict = toTypedDict<string, string>(v.toGenericDict());
          EXPECT_EQ(2, dict.size());
          EXPECT_EQ("value1", dict.at("key1"));
          EXPECT_EQ("value2", dict.at("key2"));
        },
        "(Dict(str, str) a) -> Dict(str, str)");
      unordered_map<i64, Tensor> tensor_map;
      tensor_map.emplace(1, dummyTensor(DispatchKey::CPU));
      tensor_map.emplace(2, dummyTensor(DispatchKey::CUDA));
      testArgTypes<unordered_map<i64, Tensor>>::test<TestLegacyAPI>(
        tensor_map, [] (unordered_map<i64, Tensor> v) {
          EXPECT_EQ(2, v.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(v.at(1)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(v.at(2)));
        },
        tensor_map, [] (const IValue& v) {
          Dict<i64, Tensor> dict = toTypedDict<i64, Tensor>(v.toGenericDict());
          EXPECT_EQ(2, dict.size());
          EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(dict.at(1)));
          EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
        },
        "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

      // weird deeply nested type
      using DeeplyNestedType = List<Dict<string, List<optional<Dict<i64, string>>>>>;
      auto makeDeeplyNestedObject = [] () -> DeeplyNestedType {
        Dict<i64, string> inner3;
        inner3.insert(1, "1");
        List<optional<Dict<i64, string>>> inner2;
        inner2.push_back(move(inner3));
        Dict<string, List<optional<Dict<i64, string>>>> inner1;
        inner1.insert("key", move(inner2));
        List<Dict<string, List<optional<Dict<i64, string>>>>> result;
        result.push_back(inner1);
        return result;
      };
      testArgTypes<DeeplyNestedType>::test(
        makeDeeplyNestedObject(), [] (const DeeplyNestedType& v) {EXPECT_EQ("1", v.get(0).at("key").get(0).value().at(1));},
        makeDeeplyNestedObject(), [] (const IValue& v) {EXPECT_EQ("1", v.to<DeeplyNestedType>().get(0).at("key").get(0).value().at(1));},
        "(Dict(str, Dict(int, str)?[])[] a) -> Dict(str, Dict(int, str)?[])[]");

    */
}

#[test] fn new_operator_registration_test_basics() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("dummy(Tensor self) -> Tensor");
      m.def("dummy1(Tensor self) -> Tensor");
      m.def("dummy2(Tensor self) -> Tensor");
      m.def("dummy3(Tensor self, Tensor other) -> Tensor", [](const Tensor& self, const Tensor& other) { return self; });
      m.def("dummy4", [](const Tensor& self, const Tensor& other) { return other; });
      m.impl("dummy", DeviceType_CPU, [](const Tensor& self) { return self; });
      m.impl("dummy", DeviceType_XLA, [](const Tensor& self) { return self; });
      // Internal API
      m.impl("dummy2", DispatchKey::CPU, [](const Tensor& self) { return self; });
      m.impl("dummy2", DispatchKey::XLA, [](const Tensor& self) { return self; });

      ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy", ""}).has_value());
      // Should have a schema even if there are no impls
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy1", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy2", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy3", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy4", ""}).has_value());

    */
}

#[test] fn new_operator_registration_test_import_top_level() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("def1(Tensor self) -> Tensor");
      m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
      m.def("def3", [](const Tensor& x) { return x; });

      auto m2 = MAKE_TORCH_LIBRARY_IMPL(test, CatchAll);
      m2.impl("impl1", [](const Tensor& x) { return x; });

      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def1", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def2", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def3", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());

    */
}

#[test] fn new_operator_registration_test_overload() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor self) -> Tensor");
      m.def("fn.overload1(Tensor self, Tensor other) -> Tensor");
      m.def("fn.overload2(Tensor self, Tensor other, Tensor alpha) -> Tensor");

      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload1"}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload2"}).has_value());

    */
}

#[test] fn new_operator_registration_test_import_namespace() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("def1(Tensor self) -> Tensor");
      m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
      m.def("def3", [](const Tensor& x) { return x; });
      m.impl("impl1", [](const Tensor& x) { return x; });
      expectThrows<Error>([&] {
        m.def("retest::def1(Tensor self) -> Tensor");
      }, "");

      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());

    */
}

#[test] fn new_operator_registration_test_schema() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("def1(Tensor self) -> Tensor");
      m.def(Torchschema("def2(Tensor self) -> Tensor"));
      m.def(Torchschema("def3(Tensor self) -> Tensor", AliasAnalysisKind::PURE_FUNCTION));
      m.def(TorchJitparseSchema("def4(Tensor self) -> Tensor"));

      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""}).has_value());

      EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def1", ""})->schema().aliasAnalysis(), AliasAnalysisKind::FROM_SCHEMA);
      EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def2", ""})->schema().aliasAnalysis(), AliasAnalysisKind::FROM_SCHEMA);
      EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def3", ""})->schema().aliasAnalysis(), AliasAnalysisKind::PURE_FUNCTION);
      ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""})->schema().isDefaultAliasAnalysisKind());

    */
}

#[test] fn new_operator_registration_test_when_registering_backend_fallback_kernel_and_catchall_for_same_then_calls() {
    todo!();
    /*
    
      auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
      m1.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

      bool called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor t, str input) -> ()");
      m.impl("fn", [&] (Tensor, string) { called = true; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      // CatchAll now maps to CompositeImplicitAutograd and has higher precedence than backend fallback.
      EXPECT_TRUE(called);

    */
}

#[test] fn new_operator_registration_test_when_registering_autograd_kernel_with_regular_then_can_call() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor dummy) -> ()");
      m.impl("fn", DispatchKey::CPU, nonautograd_kernel);
      m.impl("fn", DispatchKey::Autograd, autograd_kernel);

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      called_nonautograd = called_autograd = false;
      callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);

    */
}

#[test] fn new_operator_registration_test_dispatch_with_composite_implicit_autograd_kernel() {
    todo!();
    /*
    
      bool math_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        ASSERT_FALSE(math_called);
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(math_called);
      }

      {
        math_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
      }

      {
        math_called = false;
        callOp(*op, dummyTensor(DispatchKey::XLA));
        ASSERT_TRUE(math_called);
      }

      {
        math_called = false;
        callOp(*op, dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
      }

      {
        math_called = false;
        callOp(*op, dummyTensor(DispatchKey::SparseCPU));
        ASSERT_TRUE(math_called);
      }

      {
        math_called = false;
        callOp(*op, dummyTensor(DispatchKey::SparseCPU, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_with_composite_implicit_autograd_and_kernel() {
    todo!();
    /*
    
      bool math_called = false;
      bool autograd_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));
      m.impl("fn", DispatchKey::Autograd, [&](const Tensor& x) { autograd_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      // CompositeImplicitAutograd has higher precedence than Autograd
      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(autograd_called);
      }

      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(autograd_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_with_composite_implicit_autograd_and_catch_all_kernel() {
    todo!();
    /*
    
      bool math_called = false;
      bool catchall_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));
      m.impl("fn", [&](const Tensor& x) { catchall_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      // catchAll now maps to CompositeImplicitAutograd, which means we have two registrations to CompositeImplicitAutograd key.
      // The last registration is used.
      {
        catchall_called = math_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_FALSE(math_called);
        ASSERT_TRUE(catchall_called);
      }

      {
        catchall_called = math_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_FALSE(math_called);
        ASSERT_TRUE(catchall_called);
      }

    */
}

#[test] fn new_operator_registration_test_autograd_backend_overrides_composite_implicit_kernel() {
    todo!();
    /*
    
      bool math_called = false;
      bool autograd_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));
      m.impl("fn", DispatchKey::AutogradCPU, [&](const Tensor& x) { autograd_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(autograd_called);
      }

      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);
        ASSERT_FALSE(math_called);
      }

      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(autograd_called);
      }

      {
        math_called = autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(autograd_called);
      }

    */
}

#[test] fn new_operator_registration_test_backend_overrides_composite_implicit_autograd_kernel() {
    todo!();
    /*
    
      bool math_called = false;
      bool backend_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));
      m.impl("fn", DispatchKey::CPU, [&](const Tensor& x) { backend_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        math_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(math_called);
      }

      {
        // Fallthrough AutogradCPU and reaches CPU
        math_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(math_called);
      }

      {
        math_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(backend_called);
      }

      {
        math_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
        ASSERT_TRUE(math_called);
        ASSERT_FALSE(backend_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_with_composite_explicit_autograd_kernel() {
    todo!();
    /*
    
      bool called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { called = true; return x; }));

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        ASSERT_FALSE(called);
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(called);
      }

      {
        called = false;
        // AutogradCPU is fallthrough, calls CPU kernel
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(called);
      }

      {
        called = false;
        callOp(*op, dummyTensor(DispatchKey::XLA));
        ASSERT_TRUE(called);
      }

      {
        called = false;
        // AutogradXLA is fallthrough, calls XLA kernel
        callOp(*op, dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        ASSERT_TRUE(called);
      }

      {
        called = false;
        callOp(*op, dummyTensor(DispatchKey::SparseCPU));
        ASSERT_TRUE(called);
      }

      {
        called = false;
        // AutogradCPU is fallthrough, calls CPU kernel
        callOp(*op, dummyTensor(DispatchKey::SparseCPU, /*requires_grad=*/true));
        ASSERT_TRUE(called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_with_composite_explicit_autograd_and_implicit_kernel() {
    todo!();
    /*
    
      bool backend_called = false;
      bool math_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { backend_called = true; return x; }));
      m.impl("fn", DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        backend_called = math_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(math_called);
      }

      {
        backend_called = math_called = false;
        // AutogradCPU is fallthrough, calls CPU kernel
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_FALSE(math_called);
        ASSERT_TRUE(backend_called);
      }

      {
        backend_called = math_called = false;
        callOp(*op, dummyTensor(DispatchKey::XLA));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(math_called);
      }

      {
        backend_called = math_called = false;
        // AutogradXLA is fallthrough, calls XLA kernel
        callOp(*op, dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        ASSERT_FALSE(math_called);
        ASSERT_TRUE(backend_called);
      }

      {
        backend_called = math_called = false;
        callOp(*op, dummyTensor(DispatchKey::SparseCPU));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(math_called);
      }

      {
        backend_called = math_called = false;
        // AutogradOther is fallthrough, calls SparseCPU kernel
        callOp(*op, dummyTensor(DispatchKey::SparseCPU, /*requires_grad=*/true));
        ASSERT_FALSE(math_called);
        ASSERT_TRUE(backend_called);
      }

    */
}

#[test] fn new_operator_registration_test_backend_overrides_composite_explicit_autograd_kernel() {
    todo!();
    /*
    
      bool default_called = false;
      bool backend_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { default_called = true; return x; }));
      m.impl("fn", DispatchKey::CPU, [&](const Tensor& x) { backend_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        default_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(default_called);
      }

      {
        default_called = backend_called = false;
        // AutogradCPU is fallthrough, calls CPU kernel
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(backend_called);
        ASSERT_FALSE(default_called);
      }

      {
        default_called = backend_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA));
        ASSERT_TRUE(default_called);
        ASSERT_FALSE(backend_called);
      }

      {
        default_called = backend_called = false;
        // AutogradCUDA is fallthrough, calls CUDA kernel
        callOp(*op, dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
        ASSERT_TRUE(default_called);
        ASSERT_FALSE(backend_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch() {
    todo!();
    /*
    
      bool cpu_called = false;
      bool cuda_called = false;
      bool autograd_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn_cpu", Torchdispatch(DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));
      m.def("fn_cuda", Torchdispatch(kCUDA, [&](const Tensor& x) { cuda_called = true; return x; }));
      m.def("fn_autograd", Torchdispatch(kAutograd, [&](const Tensor& x) { autograd_called = true; return x; }));

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn_cpu", ""});
        ASSERT_TRUE(op.has_value());
        ASSERT_FALSE(cpu_called);
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(cpu_called);
      }

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn_cuda", ""});
        ASSERT_TRUE(op.has_value());
        ASSERT_FALSE(cuda_called);
        callOp(*op, dummyTensor(DispatchKey::CUDA));
        ASSERT_TRUE(cuda_called);
      }

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
        ASSERT_TRUE(op.has_value());
        ASSERT_FALSE(autograd_called);
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);
      }

      {
        autograd_called = false;
        auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
        ASSERT_TRUE(op.has_value());
        callOp(*op, dummyTensor(DispatchKey::XLA, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_autograd_precedence() {
    todo!();
    /*
    
      bool cpu_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        ASSERT_FALSE(cpu_called);
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(cpu_called);
      }

      {
        // AutogradCPU is fallthrough, use CPU kernel
        cpu_called = false;
        auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(cpu_called);
      }

      bool autograd_called = false;
      m.impl("fn", kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);
      }

      // Autograd backend kernel has higher precedence than Autograd alias.
      bool autogradcpu_called = false;
      m.impl("fn", DispatchKey::AutogradCPU, [&](const Tensor& x) { autogradcpu_called = true; return x; });

      {
        auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(autogradcpu_called);
      }

    */
}

#[test] fn new_operator_registration_test_throws_when_register_to_backend_maps_autograd_other() {
    todo!();
    /*
    
      bool sparsecpu_called, math_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::SparseCPU, [&](const Tensor& x) { sparsecpu_called = true; return x; }));
      m.impl("fn", DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        callOp(*op, dummyTensor(DispatchKey::SparseCPU));
        ASSERT_TRUE(sparsecpu_called);
      }

      {
        expectThrows<Error>([&] {
          callOp(*op, dummyTensor(DispatchKey::SparseCPU, /*requires_grad=*/true));
        }, "test::fn has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther.");
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_multiple_tensors() {
    todo!();
    /*
    
      bool privateuse1_called = false;
      bool catchall_called = false;
      // Similar to in-tree AutogradCPU/AutogradCUDA etc, out-of-tree backends usually register
      // a fallthrough kernel for AutogradPrivateUse1.
      auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1);
      m1.fallback(CppFunction::makeFallthrough());

      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn", Torchdispatch(DispatchKey::PrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; }));
      m.impl("fn", [&](const Tensor& x, const Tensor& y) { catchall_called = true; return x; });

      {
        auto op = Dispatcher::singleton().findOp({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        callOp(*op, dummyTensor(DispatchKey::PrivateUse1), dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(privateuse1_called);
      }

      {
        auto op = Dispatcher::singleton().findOp({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        ASSERT_FALSE(catchall_called);
        callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(catchall_called);
      }

      {
        auto op = Dispatcher::singleton().findOp({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        catchall_called = false;
        callOp(*op,
               dummyTensor(DispatchKey::CPU, /*requires_grad=*/true),
               dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(catchall_called);
      }

      {
        // TODO(#43908): currently this will fallthrough AutogradPrivateUse1 then call catchall kernel
        // at AutogradCPU, while backend extenders are indeed expecting to call PrivateUse1 kernel.
        // This confusing behavior is caused by we registering fallthrough as backend fallback for
        // Autograd keys. Note users could always work around this by registering the same kernel to
        // AutogradPrivateUse1 as shown below until we support it.
        auto op = Dispatcher::singleton().findOp({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        catchall_called = false;
        callOp(*op,
               dummyTensor(DispatchKey::PrivateUse1, /*requires_grad=*/true),
               dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(catchall_called);
      }

      m.impl("fn", DispatchKey::AutogradPrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; });

      {
        auto op = Dispatcher::singleton().findOp({"test::fn", ""});
        ASSERT_TRUE(op.has_value());
        privateuse1_called = false;
        callOp(*op,
               dummyTensor(DispatchKey::PrivateUse1, /*requires_grad=*/true),
               dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(privateuse1_called);
      }

    */
}

#[test] fn new_operator_registration_test_dispatch_multiple() {
    todo!();
    /*
    
      bool cpu_called = false;
      bool cuda_called = false;
      bool autograd_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor self) -> Tensor");
      // NB: Direct use of DispatchKey is discouraged; use the DeviceType
      // k-synonyms instead
      m.impl("fn", DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; });
      m.impl("fn", kCUDA, [&](const Tensor& x) { cuda_called = true; return x; });
      m.impl("fn", kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      {
        ASSERT_FALSE(cpu_called);
        callOp(*op, dummyTensor(DispatchKey::CPU));
        ASSERT_TRUE(cpu_called);

        ASSERT_FALSE(cuda_called);
        callOp(*op, dummyTensor(DispatchKey::CUDA));
        ASSERT_TRUE(cuda_called);
      }

      {
        ASSERT_FALSE(autograd_called);
        callOp(*op, dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);

        autograd_called = false;
        callOp(*op, dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
        ASSERT_TRUE(autograd_called);
      }

    */
}

#[test] fn new_operator_registration_test_fallback() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
      m.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

      auto registrar1 = RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");

      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());
      auto stack = callOp(*op, dummyTensor(DispatchKey::CPU), "hello ");
      EXPECT_EQ("hello _test::dummy", stack[1].toString()->string());

    */
}

#[test] fn new_operator_registration_test_backend_select_redispatches_to_cpu() {
    todo!();
    /*
    
      bool cpu_called = false;
      bool backend_generic_called = false;
      auto m = MAKE_TORCH_LIBRARY(test);
      auto after_backend_select = DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);
      m.def("fn(Tensor self) -> Tensor");
      m.impl("fn", kCPU, [&](const Tensor& x) { cpu_called = true; return x; });
      m.impl("fn", DispatchKey::BackendSelect, [&](DispatchKeySet ks, const Tensor& x) {
         backend_generic_called = true;
         auto op = Dispatcher::singleton().findSchema({"test::fn", ""}).value().typed<Tensor (const Tensor&)>();
         return Dispatcher::singleton().redispatch<Tensor, const Tensor&>(op, ks & after_backend_select, x);
       });

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());
      callOp(*op, dummyTensor(DispatchKey::CPU));
      ASSERT_TRUE(cpu_called);
      ASSERT_TRUE(backend_generic_called);

    */
}

#[test] fn new_operator_registration_test_torch_library_twice_is_error() {
    todo!();
    /*
    
      {
        auto m = MAKE_TORCH_LIBRARY(test);
        expectThrows<Error>([] {
          auto m2 = MAKE_TORCH_LIBRARY(test);
        }, "Only a single TORCH_LIBRARY");
      }
      // Ensure it's ok after deregistering
      auto m = MAKE_TORCH_LIBRARY(test);

    */
}

pub fn dummy_fn(x: &Tensor) -> Tensor {
    
    todo!();
        /*
            return x;
        */
}

#[test] fn new_operator_registration_test_cpp_function() {
    todo!();
    /*
    
      // Just show off the possible ways to register functions
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn1", &dummy_fn);
      // C++ will implicitly convert function to function pointer
      // c.f. https://en.cppreference.com/w/cpp/language/implicit_conversion#Function_to_pointer
      m.def("fn2", dummy_fn);
      m.def("fn3", [](const Tensor& x) { return x; });
      // These require explicit schema
      m.def("fn4(Tensor x) -> Tensor", CppFunction::makeFallthrough());
      m.def("fn5(Tensor x) -> Tensor", CppFunction::makeFromUnboxedFunction(dummy_fn));
      m.def("fn6(Tensor x) -> Tensor", CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

    */
}

/**
  | Some internal tests that have to be done
  | from C++
  |
  */
pub struct OpRegistrationListenerForDelayedListenerTest {
    base:            OpRegistrationListener,
    num_registers:   i64, // default = 0
    num_deregisters: i64, // default = 0
}

impl OpRegistrationListenerForDelayedListenerTest {
    
    pub fn on_operator_registered(&mut self, op: &OperatorHandle)  {
        
        todo!();
        /*
            num_registers_++;
        */
    }
    
    pub fn on_operator_deregistered(&mut self, op: &OperatorHandle)  {
        
        todo!();
        /*
            num_deregisters_++;
        */
    }
}

#[test] fn new_operator_registration_test_delayed_listener() {
    todo!();
    /*
    
      auto listener = make_unique<OpRegistrationListenerForDelayedListenerTest>();
      auto listener_ptr = listener.get();
      auto registry = Dispatcher::singleton().addRegistrationListener(move(listener));
      i64 initial_num_registers = listener_ptr->num_registers_;
      i64 initial_num_deregisters = listener_ptr->num_deregisters_;
      auto op = Dispatcher::singleton().findOp({"_test::dummy", ""});
      ASSERT_FALSE(op.has_value());
      auto m1 = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
      m1.impl("dummy", [](const Tensor& self) { return self; });
      EXPECT_EQ(initial_num_registers, listener_ptr->num_registers_);
      {
        auto m2 = MAKE_TORCH_LIBRARY(_test);
        m2.def("dummy(Tensor self) -> Tensor");
        EXPECT_EQ(initial_num_registers + 1, listener_ptr->num_registers_);
      }
      EXPECT_EQ(initial_num_deregisters + 1, listener_ptr->num_deregisters_);

    */
}

#[test] fn new_operator_registration_test_impl_no_def_gets_caught() {
    todo!();
    /*
    
      auto danglingImpls = Dispatcher::singleton().findDanglingImpls();
      string error_str = "Discovered operators that have been registered through the dispatcher"
                              " without explicitly specifying their schemas. Please do so using"
                              " the TORCH_LIBRARY macro. Suspect operators:\n";
      for (auto& op : danglingImpls) {
          auto& op_name = op.operator_name();
          error_str += "\t" + op_name.name;
          if (op_name.overload_name != "") {
              error_str += "." + op_name.overload_name;
          }
          error_str += "\n";
      }
      ASSERT_EQ(danglingImpls.size(), 0) << error_str;

    */
}

lazy_static!{
    /*
    bool called_kernel_cpu = false;
    bool called_kernel_autograd = false;
    bool called_kernel_tracing = false;
    */
}

pub fn cpu_kernel(_0: Tensor)  {
    
    todo!();
        /*
            called_kernel_cpu = true;
        */
}

/**
  | autograd kernel that redispatches.
  | Explicitly takes in and updates the
  | DispatchKeySet
  |
  */
pub fn autograd_kernel_redispatching_with_dispatch_key_set(
        ks: DispatchKeySet,
        a:  Tensor)  {
    
    todo!();
        /*
            called_kernel_autograd = true;
      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      auto updatedDispatchKeySet = ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::AutogradOther);
      callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
        */
}

/**
  | autograd kernel that redispatches.
  | Does not take in a DispatchKeySet
  |
  */
pub fn autograd_kernel_redispatching_without_dispatch_key_set(
        ks: DispatchKeySet,
        a:  Tensor)  {
    
    todo!();
        /*
            called_kernel_autograd = true;
      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      auto updatedDispatchKeySet = ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::AutogradOther);
      callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
        */
}

/**
  | tracing kernel that redispatches.
  | Explicitly takes in and updates the
  | DispatchKeySet
  |
  */
pub fn tracing_kernel_redispatching_with_dispatch_key_set(
        ks: DispatchKeySet,
        a:  Tensor)  {
    
    todo!();
        /*
            called_kernel_tracing = true;
      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      auto updatedDispatchKeySet = ks & DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::Tracer);
      callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
        */
}

#[test] fn operator_registration_test_call_kernels_with_dispatch_key_set_convention_redispatches_to_lower_priority() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor dummy) -> ()");
      m.impl("fn", DispatchKey::CPU, cpu_kernel);
      m.impl("fn", DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
      m.impl("fn", DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
      auto tracing_autograd_cpu_set = DispatchKeySet()
                                        .add(DispatchKey::Tracer)
                                        .add(DispatchKey::AutogradCPU)
                                        .add(DispatchKey::CPU);

      // call Tracing -> call Autograd -> call CPU
      callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
      EXPECT_TRUE(called_kernel_tracing);
      EXPECT_TRUE(called_kernel_autograd);
      EXPECT_TRUE(called_kernel_cpu);

    */
}

#[test] fn operator_registration_test_call_kernels_with_dispatch_key_set_convention_boxed_redispatches_to_lower_priority() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor dummy) -> ()");
      m.impl("fn", DispatchKey::CPU, cpu_kernel);
      m.impl("fn", DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
      m.impl("fn", DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
      auto tracing_autograd_cpu_set = DispatchKeySet()
                                        .add(DispatchKey::Tracer)
                                        .add(DispatchKey::AutogradCPU)
                                        .add(DispatchKey::CPU);

      // call Tracing -> call Autograd -> call CPU
      callOp<Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
      EXPECT_TRUE(called_kernel_tracing);
      EXPECT_TRUE(called_kernel_autograd);
      EXPECT_TRUE(called_kernel_cpu);

    */
}

#[test] fn operator_registration_test_call_kernels_with_dispatch_key_set_convention_mixed_calling_conventions_redispatches_to_lower_priority() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(test);
      m.def("fn(Tensor dummy) -> ()");
      m.impl("fn", DispatchKey::CPU, cpu_kernel);
      // the tracing kernel takes in a DispatchKeySet, but the autograd kernel does not
      // the dispatcher should handle correctly plumbing its DispatchKeySet to tracing and not autograd.
      m.impl("fn", DispatchKey::AutogradCPU, autograd_kernel_redispatching_without_DispatchKeySet);
      m.impl("fn", DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

      auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
      ASSERT_TRUE(op.has_value());

      called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
      auto tracing_autograd_cpu_set = DispatchKeySet()
                                        .add(DispatchKey::Tracer)
                                        .add(DispatchKey::AutogradCPU)
                                        .add(DispatchKey::CPU);

      // call Tracing -> call Autograd -> call CPU
      callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
      EXPECT_TRUE(called_kernel_tracing);
      EXPECT_TRUE(called_kernel_autograd);
      EXPECT_TRUE(called_kernel_cpu);

    */
}
