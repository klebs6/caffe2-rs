crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/KernelFunction_test.cpp]

/**
  | This namespace contains several fake kernels.
  |
  | Some of these kernels expect to be called with
  | two i64 arguments and store these arguments
  | in called_with_args.
  |
  | Kernels may return a single value, or multiple
  | values, or no value.
  |
  | The kernels with a single return value return
  | int value 5,
  |
  | The expectXXX() functions further below use
  | these invariants to check that calling
  | a specific kernels works correctly.
  */
pub mod kernels {

    use super::*;

    lazy_static!{
        /*
        optional<tuple<i64, i64>> called_with_args;
        */
    }

    /**
      | The calling convention in the dispatcher
      | requires that calls to
      | KernelFunction::call()/callBoxed() take in
      | a DispatchKeySet.
      |
      | The value itself is meaningless for all of
      | the tests that use kernels without
      | a DispatchKeySet argument.
      |
      | See Note [Plumbing Keys Through The
      | Dispatcher] for details.
      |
      */
    lazy_static!{
        /*
        c10::DispatchKeySet CPU_TEST_SET = c10::DispatchKeySet(c10::DispatchKey::CPU);
        */
    }

    pub fn boxed_func_with_return(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                EXPECT_EQ(2, stack->size());
              EXPECT_TRUE(stack->at(0).isInt());
              EXPECT_TRUE(stack->at(1).isInt());
              called_with_args = tuple<i64, i64>(stack->at(0).toInt(), stack->at(1).toInt());

              stack->clear();
              stack->push_back(5);
            */
    }

    pub fn boxed_func_without_return(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                EXPECT_EQ(2, stack->size());
              EXPECT_TRUE(stack->at(0).isInt());
              EXPECT_TRUE(stack->at(1).isInt());
              called_with_args = tuple<i64, i64>(stack->at(0).toInt(), stack->at(1).toInt());

              stack->clear();
            */
    }

    pub fn boxed_func_with_multi_return(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                EXPECT_EQ(2, stack->size());
              EXPECT_TRUE(stack->at(0).isInt());
              i64 a = stack->at(0).toInt();
              EXPECT_TRUE(stack->at(1).isInt());
              i64 b = stack->at(1).toInt();
              called_with_args = tuple<i64, i64>(a, b);

              stack->clear();
              torch::jit::push(stack, a + b);
              torch::jit::push(stack, a * b);
            */
    }

    lazy_static!{
        /*
        struct unboxed_functor_with_return final : OperatorKernel {
              i64 operator()(i64 a, i64 b) {
                called_with_args = tuple<i64, i64>(a, b);
                return 5;
              }
            };

            struct unboxed_functor_without_return final : OperatorKernel {
              void operator()(i64 a, i64 b) {
                called_with_args = tuple<i64, i64>(a, b);
              }
            };

            struct unboxed_functor_with_return_factory final {
              std::unique_ptr<OperatorKernel> operator()() {
                return std::make_unique<unboxed_functor_with_return>();
              }
            };

            struct unboxed_functor_without_return_factory final {
              std::unique_ptr<OperatorKernel> operator()() {
                return std::make_unique<unboxed_functor_without_return>();
              }
            };
        */
    }

    pub fn unboxed_function_with_return(a: i64, b: i64) -> i64 {
        
        todo!();
            /*
                called_with_args = tuple<i64, i64>(a, b);
              return 5;
            */
    }

    pub fn unboxed_function_without_return(a: i64, b: i64)  {
        
        todo!();
            /*
                called_with_args = tuple<i64, i64>(a, b);
            */
    }

    lazy_static!{
        /*
        auto unboxed_lambda_with_return = [] (i64 a, i64 b) -> i64 {
              called_with_args = tuple<i64, i64>(a, b);
              return 5;
            };

            auto unboxed_lambda_without_return = [] (i64 a, i64 b) -> void{
              called_with_args = tuple<i64, i64>(a, b);
            };
        */
    }

    pub fn make_dummy_operator_handle() -> OperatorHandle {
        
        todo!();
            /*
                static auto registry = torch::RegisterOperators().op("my::dummy() -> ()");
          return c10::Dispatcher::singleton().findSchema({"my::dummy", ""}).value();
            */
    }

    /**
      | boxed kernels that return refs to tensor
      | arguments, a la inplace/outplace kernels
      |
      */
    pub fn boxed_func_for_inplace_op(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                // (Tensor(a!), Scalar) -> Tensor(a!)
          EXPECT_EQ(2, stack->size());

          ASSERT_TRUE(stack->at(0).isTensor());
          auto t = stack->at(0).toTensor();

          ASSERT_TRUE(stack->at(1).isScalar());
          auto s = stack->at(1).toScalar();

          t.add_(s);

          stack->clear();
          torch::jit::push(stack, t);
            */
    }

    pub fn boxed_func_for_outofplace_op(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                // (Scalar, Tensor(a!)) -> Tensor(a!)
          EXPECT_EQ(2, stack->size());

          ASSERT_TRUE(stack->at(0).isScalar());
          auto s = stack->at(0).toScalar();

          ASSERT_TRUE(stack->at(1).isTensor());
          auto t = stack->at(1).toTensor();

          t.add_(s);

          stack->clear();
          torch::jit::push(stack, t);
            */
    }

    pub fn boxed_func_for_outofplace_multi_op(
            op_handle: &OperatorHandle,
            stack:     *mut Stack)  {
        
        todo!();
            /*
                // (Scalar, Scalar, Tensor(a!), Tensor(b!)) -> (Tensor(a!), Tensor(b!))
          EXPECT_EQ(4, stack->size());

          ASSERT_TRUE(stack->at(0).isScalar());
          auto s1 = stack->at(0).toScalar();

          ASSERT_TRUE(stack->at(1).isScalar());
          auto s2 = stack->at(1).toScalar();

          ASSERT_TRUE(stack->at(2).isTensor());
          auto t1 = stack->at(2).toTensor();

          ASSERT_TRUE(stack->at(3).isTensor());
          auto t2 = stack->at(3).toTensor();

          t1.add_(s1);
          t2.add_(s2);

          stack->clear();
          torch::jit::push(stack, t1);
          torch::jit::push(stack, t2);
            */
    }

    /* ------------- boxed calling tests:  ------------- */

    // functional
    pub fn expect_boxed_calling_with_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          vector<IValue> stack {3, 4};
          OperatorHandle dummy = makeDummyOperatorHandle();

          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);
          EXPECT_EQ(1, stack.size());
          EXPECT_TRUE(stack[0].isInt());
          EXPECT_EQ(5, stack[0].toInt());
            */
    }

    pub fn expect_boxed_calling_without_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          vector<IValue> stack {3, 4};
          OperatorHandle dummy = makeDummyOperatorHandle();

          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);
          EXPECT_EQ(0, stack.size());
            */
    }

    pub fn expect_boxed_calling_with_multi_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          vector<IValue> stack {3, 4};
          OperatorHandle dummy = makeDummyOperatorHandle();

          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);
          EXPECT_EQ(2, stack.size());

          EXPECT_TRUE(stack[0].isInt());
          EXPECT_EQ(7, stack[0].toInt());

          EXPECT_TRUE(stack[1].isInt());
          EXPECT_EQ(12, stack[1].toInt());
            */
    }

    // in/out
    pub fn expect_in_place_boxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto t = at::zeros({1});
          auto s = 1.0f;
          vector<IValue> stack {t, s};
          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          // kernel should have updated out arg and returned it
          EXPECT_EQ(t.item().toFloat(), 1.0f);
          EXPECT_EQ(1, stack.size());
          EXPECT_TRUE(stack[0].isTensor());
          EXPECT_TRUE(stack[0].toTensor().is_same(t));
            */
    }

    pub fn expect_out_of_place_boxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto s = 1.0f;
          auto t = at::zeros({1});
          vector<IValue> stack {s, t};
          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          // kernel should have updated out arg and returned it on the stack
          EXPECT_EQ(t.item().toFloat(), 1.0f);
          EXPECT_EQ(1, stack.size());
          EXPECT_TRUE(stack[0].isTensor());
          EXPECT_TRUE(stack[0].toTensor().is_same(t));
            */
    }

    pub fn expect_out_of_place_multi_boxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto s1 = 1.0f;
          auto s2 = 2.0f;
          auto t1 = at::zeros({1});
          auto t2 = at::zeros({1});
          vector<IValue> stack {s1, s2, t1, t2};
          func.callBoxed(dummy, CPU_TEST_SET, &stack);

          // kernel should have updated output args and returned them on the stack
          EXPECT_EQ(t1.item().toFloat(), 1.0f);
          EXPECT_EQ(t2.item().toFloat(), 2.0f);
          EXPECT_EQ(2, stack.size());
          EXPECT_TRUE(stack[0].isTensor());
          EXPECT_TRUE(stack[0].toTensor().is_same(t1));
          EXPECT_TRUE(stack[1].isTensor());
          EXPECT_TRUE(stack[1].toTensor().is_same(t2));
            */
    }

    pub fn expect_boxed_calling_fails_with(
            func:          &KernelFunction,
            error_message: *const u8)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          vector<IValue> stack {3, 4};
          OperatorHandle dummy = makeDummyOperatorHandle();

          expectThrows<c10::Error>([&] {
            func.callBoxed(dummy, CPU_TEST_SET, &stack);
          }, errorMessage);
            */
    }

    /* ------------ unboxed calling tests:  ------------ */

    // functional

    /**
      | make an unboxed call to a kernel that
      | returns a single value.
      |
      */
    pub fn expect_unboxed_calling_with_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          OperatorHandle dummy = makeDummyOperatorHandle();

          i64 result = func.call<i64, i64, i64>(dummy, CPU_TEST_SET, 3, 4);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);
          EXPECT_EQ(5, result);
            */
    }

    /**
      | make an unboxed call to a kernel that
      | returns nothing.
      |
      */
    pub fn expect_unboxed_calling_without_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          OperatorHandle dummy = makeDummyOperatorHandle();

          func.call<void, i64, i64>(dummy, CPU_TEST_SET, 3, 4);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);
            */
    }

    /**
      | make an unboxed call to a kernel that returns
      | two values.
      |
      | When calling unboxed, multiple values are
      | returned as a tuple.
      |
      */
    pub fn expect_unboxed_calling_with_multi_return_works(func: &KernelFunction)  {
        
        todo!();
            /*
                called_with_args = c10::nullopt;
          OperatorHandle dummy = makeDummyOperatorHandle();

          auto result = func.call<std::tuple<i64, i64>, i64, i64>(dummy, CPU_TEST_SET, 3, 4);

          EXPECT_TRUE(called_with_args.has_value());
          EXPECT_EQ((tuple<i64, i64>(3, 4)), *called_with_args);

          EXPECT_EQ((tuple<i64, i64>(7, 12)), result);
            */
    }

    // in/out
    pub fn expect_in_place_unboxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto t = at::zeros({1});
          at::Tensor& t_out = func.call<at::Tensor&, at::Tensor&, at::Scalar>(dummy, CPU_TEST_SET, t, 1.0f);

          // should have updated first arg and returned it
          EXPECT_EQ(t.item().toFloat(), 1.0f);
          EXPECT_EQ(&t, &t_out);
            */
    }

    pub fn expect_out_of_place_unboxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto t = at::zeros({1});
          at::Tensor& t_out = func.call<at::Tensor&, at::Scalar, at::Tensor&>(dummy, CPU_TEST_SET, 1.0f, t);

          // should have updated out arg and returned it
          EXPECT_EQ(t.item().toFloat(), 1.0f);
          EXPECT_EQ(&t, &t_out);
            */
    }

    pub fn expect_out_of_place_multi_unboxed_calling_works(func: &KernelFunction)  {
        
        todo!();
            /*
                OperatorHandle dummy = makeDummyOperatorHandle();

          auto s1 = 1.0f;
          auto s2 = 2.0f;
          auto t1 = at::zeros({1});
          auto t2 = at::zeros({1});

          std::tuple<at::Tensor&, at::Tensor&> tup = func.call<
            std::tuple<at::Tensor&, at::Tensor&>, at::Scalar, at::Scalar, at::Tensor&, at::Tensor&
          >(dummy, CPU_TEST_SET, s1, s2, t1, t2);

          // kernel should have updated out args and returned them in a tuple
          EXPECT_EQ(t1.item().toFloat(), 1.0f);
          EXPECT_EQ(t2.item().toFloat(), 2.0f);

          auto t1_out = std::get<0>(tup);
          EXPECT_EQ(t1_out.item().toFloat(), 1.0f);
          EXPECT_TRUE(t1_out.is_same(t1));

          auto t2_out = std::get<1>(tup);
          EXPECT_EQ(t2_out.item().toFloat(), 2.0f);
          EXPECT_TRUE(t2_out.is_same(t2));
            */
    }
}

/* ----------- functional, boxed calling  ----------- */

#[test] fn kernel_function_test_given_boxed_with_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
      kernels::expectBoxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_without_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
      kernels::expectBoxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_multi_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
      kernels::expectBoxedCallingWithMultiReturnWorks(func);

    */
}

/* ------------- in/out, boxed calling  ------------- */

#[test] fn kernel_function_test_given_boxed_with_in_place_signature_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
      kernels::expectInPlaceBoxedCallingWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_out_of_place_signature_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
      kernels::expectOutOfPlaceBoxedCallingWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_out_of_place_multi_signature_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
      kernels::expectOutOfPlaceMultiBoxedCallingWorks(func);

    */
}

/* ---------- functional, unboxed calling  ---------- */

#[test] fn kernel_function_test_given_boxed_with_return_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
      kernels::expectUnboxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_without_return_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
      kernels::expectUnboxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_multi_return_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
      kernels::expectUnboxedCallingWithMultiReturnWorks(func);

    */
}

/* ------------ in/out, unboxed calling  ------------ */

#[test] fn kernel_function_test_given_boxed_with_in_place_signature_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
      kernels::expectInPlaceUnboxedCallingWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_out_of_place_signature_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
      kernels::expectOutOfPlaceUnboxedCallingWorks(func);

    */
}

#[test] fn kernel_function_test_given_boxed_with_out_of_place_multi_signature_when_calling_unboxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
      kernels::expectOutOfPlaceMultiUnboxedCallingWorks(func);

    */
}

/* ----------------- functors etc.  ----------------- */

#[test] fn kernel_function_test_given_unboxed_functor_with_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
      kernels::expectBoxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_functor_without_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
      kernels::expectBoxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_functor_with_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
      kernels::expectUnboxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_functor_without_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
      kernels::expectUnboxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_with_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
      kernels::expectBoxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_without_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
      kernels::expectBoxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_with_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
      kernels::expectUnboxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_without_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
      kernels::expectUnboxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_runtime_with_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
      kernels::expectBoxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_runtime_without_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
      kernels::expectBoxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_runtime_with_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
      kernels::expectUnboxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_runtime_without_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
      kernels::expectUnboxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_lambda_with_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
      kernels::expectBoxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_lambda_without_return_when_calling_boxed_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
      kernels::expectBoxedCallingWithoutReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_lambda_with_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
      kernels::expectUnboxedCallingWithReturnWorks(func);

    */
}

#[test] fn kernel_function_test_given_unboxed_lambda_without_return_when_calling_then_works() {
    todo!();
    /*
    
      KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
      kernels::expectUnboxedCallingWithoutReturnWorks(func);

    */
}

/*
  | TODO Also test different variants of
  | calling unboxed with wrong signatures
  |
  */
