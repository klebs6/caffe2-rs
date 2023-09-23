crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/kernel_function_test.cpp]

pub fn error_kernel(
        tensor: &Tensor,
        input:  i64) -> i64 {
    
    todo!();
        /*
            EXPECT_TRUE(false); // this kernel should never be called
      return 0;
        */
}

pub fn increment_kernel(
        tensor: &Tensor,
        input:  i64) -> i64 {
    
    todo!();
        /*
            return input + 1;
        */
}

pub fn decrement_kernel(
        tensor: &Tensor,
        input:  i64) -> i64 {
    
    todo!();
        /*
            return input - 1;
        */
}

pub fn expect_calls_increment(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      auto result = callOp(*op, dummyTensor(dispatch_key), 5);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(6, result[0].toInt());
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

#[test] fn operator_registration_test_function_based_kernel_given_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_when_registered_with_torch_library_and_fn_then_can_be_called() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("my_op(Tensor dummy, int input) -> int");
      m.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_catch_all_when_registered_with_torch_library_and_fn_then_can_be_called() {
    todo!();
    /*
    
      auto m = MAKE_TORCH_LIBRARY(_test);
      m.def("my_op(Tensor dummy, int input) -> int", TORCH_FN(incrementKernel));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_multiple_operators_and_kernels_when_registered_in_one_registrar_then_calls_right() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
                                                                                          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA))
          .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
                                                                                          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_multiple_operators_and_kernels_when_registered_in_registrars_then_calls_right() {
    todo!();
    /*
    
      auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
      auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn new_operator_registration_test_function_based_kernel_given_when_runs_out_of_scope_then_cannot_be_called_anymore() {
    todo!();
    /*
    
      {
        auto m = MAKE_TORCH_LIBRARY(_test);
        m.def("_test::my_op(Tensor dummy, int input) -> int");
        auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
        m_cpu.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
        {
          auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
          m_cuda.impl("my_op", DispatchKey::CUDA, TORCH_FN(decrementKernel));

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
    bool was_called = false;
    */
}

pub fn kernel_without_output(_0: &Tensor)  {
    
    todo!();
        /*
            was_called = true;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithoutOutput), &kernelWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

pub fn kernel_with_zero_outputs(_0: &Tensor) -> () {
    
    todo!();
        /*
            was_called = true;
      return std::make_tuple();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_zero_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithZeroOutputs), &kernelWithZeroOutputs>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

pub fn kernel_with_int_output(
        _0: Tensor,
        a:  i64,
        b:  i64) -> i64 {
    
    todo!();
        /*
            return a + b;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_output(Tensor dummy, int a, int b) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntOutput), &kernelWithIntOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(9, result[0].toInt());

    */
}

pub fn kernel_with_tensor_output(input: &Tensor) -> Tensor {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CPU)
                                                                                             .kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

      result = callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));

    */
}

pub fn kernel_with_tensor_list_output(
        input1: &Tensor,
        input2: &Tensor,
        input3: &Tensor) -> List<Tensor> {
    
    todo!();
        /*
            return c10::List<Tensor>({input1, input2, input3});
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
        auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<decltype(kernelWithTensorListOutput), &kernelWithTensorListOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(3, result[0].toTensorVector().size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));

    */
}

pub fn kernel_with_int_list_output(
        _0:     &Tensor,
        input1: i64,
        input2: i64,
        input3: i64) -> List<i64> {
    
    todo!();
        /*
            return c10::List<i64>({input1, input2, input3});
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<decltype(kernelWithIntListOutput), &kernelWithIntListOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(3, result[0].toIntVector().size());
      EXPECT_EQ(2, result[0].toIntVector()[0]);
      EXPECT_EQ(4, result[0].toIntVector()[1]);
      EXPECT_EQ(6, result[0].toIntVector()[2]);

    */
}

pub fn kernel_with_multiple_outputs(_0: Tensor) -> (Tensor,i64,List<Tensor>,Option<i64>,Dict<String,Tensor>) {
    
    todo!();
        /*
            Dict<string, Tensor> dict;
      dict.insert("first", dummyTensor(DispatchKey::CPU));
      dict.insert("second", dummyTensor(DispatchKey::CUDA));
      return std::tuple<Tensor, i64, c10::List<Tensor>, c10::optional<i64>, Dict<string, Tensor>>(
        dummyTensor(DispatchKey::CUDA),
        5,
        c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),
        c10::optional<i64>(c10::in_place, 0),
        dict
      );
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
         .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<decltype(kernelWithMultipleOutputs), &kernelWithMultipleOutputs>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(5, result.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
      EXPECT_EQ(5, result[1].toInt());
      EXPECT_EQ(2, result[2].toTensorVector().size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));
      EXPECT_EQ(0, result[3].toInt());
      auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
      EXPECT_EQ(2, result_dict.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));

    */
}

pub fn kernel_with_tensor_input_by_reference_with_output(input1: &Tensor) -> Tensor {
    
    todo!();
        /*
            return input1;
        */
}

pub fn kernel_with_tensor_input_by_value_with_output(input1: Tensor) -> Tensor {
    
    todo!();
        /*
            return input1;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_input_by_reference_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CPU)
                                                                                         .kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

      result = callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_input_by_value_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CPU)
                                                                                         .kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

      result = callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));

    */
}

lazy_static!{
    /*
    Tensor captured_input;
    */
}

pub fn kernel_with_tensor_input_by_reference_without_output(input1: &Tensor)  {
    
    todo!();
        /*
            captured_input = input1;
        */
}

pub fn kernel_with_tensor_input_by_value_without_output(input1: Tensor)  {
    
    todo!();
        /*
            captured_input = input1;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_input_by_reference_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CPU)
                                                                                     .kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

      outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_input_by_value_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CPU)
                                                                                     .kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CUDA));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

      outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));

    */
}

lazy_static!{
    /*
    i64 captured_int_input = 0;
    */
}

pub fn kernel_with_int_input_without_output(
        _0:     Tensor,
        input1: i64)  {
    
    todo!();
        /*
            captured_int_input = input1;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithIntInputWithoutOutput), &kernelWithIntInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_int_input = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_int_input);

    */
}

pub fn kernel_with_int_input_with_output(
        _0:     Tensor,
        input1: i64) -> i64 {
    
    todo!();
        /*
            return input1 + 1;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntInputWithOutput), &kernelWithIntInputWithOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(4, outputs[0].toInt());

    */
}

lazy_static!{
    /*
    i64 captured_input_list_size = 0;
    */
}

pub fn kernel_with_int_list_input_without_output(
        _0:     Tensor,
        input1: &List<i64>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithoutOutput), &kernelWithIntListInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_input_list_size);

    */
}

pub fn kernel_with_int_list_input_with_output(
        _0:     Tensor,
        input1: &List<i64>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_int_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithOutput), &kernelWithIntListInputWithOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(3, outputs[0].toInt());

    */
}

pub fn kernel_with_tensor_list_input_without_output(input1: &List<Tensor>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithoutOutput), &kernelWithTensorListInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

pub fn kernel_with_tensor_list_input_with_output(input1: &List<Tensor>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_tensor_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithOutput), &kernelWithTensorListInputWithOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(2, outputs[0].toInt());

    */
}

lazy_static!{
    /*
    int captured_dict_size = 0;
    */
}

pub fn kernel_with_dict_input_without_output(input1: Dict<String,Tensor>)  {
    
    todo!();
        /*
            captured_dict_size = input1.size();
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_dict_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithoutOutput), &kernelWithDictInputWithoutOutput>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_dict_size = 0;
      Dict<string, Tensor> dict;
      dict.insert("key1", dummyTensor(DispatchKey::CPU));
      dict.insert("key2", dummyTensor(DispatchKey::CUDA));
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_dict_size);

    */
}

pub fn kernel_with_dict_input_with_output(input1: Dict<String,String>) -> String {
    
    todo!();
        /*
            return input1.at("key2");
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_dict_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithOutput), &kernelWithDictInputWithOutput>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
      ASSERT_TRUE(op.has_value());

      Dict<string, string> dict;
      dict.insert("key1", "value1");
      dict.insert("key2", "value2");
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ("value2", outputs[0].toString()->string());

    */
}

pub fn kernel_with_dict_output(input: Dict<String,String>) -> Dict<String,String> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_dict_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictOutput), &kernelWithDictOutput>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
      ASSERT_TRUE(op.has_value());

      Dict<string, string> dict;
      dict.insert("key1", "value1");
      dict.insert("key2", "value2");
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(1, outputs.size());
      auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());

      EXPECT_EQ(2, output.size());
      EXPECT_EQ("value1", output.at("key1"));
      EXPECT_EQ("value2", output.at("key2"));

    */
}

lazy_static!{
    /*
    bool called = false;
    */
}

pub fn kernel_without_inputs()  {
    
    todo!();
        /*
            called = true;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_fallback_without_any_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutInputs), &kernelWithoutInputs>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op);
      EXPECT_TRUE(called);

    */
}

pub fn kernel_without_tensor_inputs(arg: i64) -> i64 {
    
    todo!();
        /*
            return arg + 1;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_fallback_without_tensor_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutTensorInputs), &kernelWithoutTensorInputs>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, 3);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(4, outputs[0].toInt());

    */
}

lazy_static!{
    /*
    c10::optional<Tensor> called_arg2 = c10::nullopt;
    c10::optional<i64> called_arg3 = c10::nullopt;
    c10::optional<std::string> called_arg4 = c10::nullopt;
    */
}

pub fn kernel_with_opt_input_without_output(
        arg1: Tensor,
        arg2: &Option<Tensor>,
        arg3: Option<i64>,
        arg4: Option<String>)  {
    
    todo!();
        /*
            called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_optional_inputs_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithoutOutput), &kernelWithOptInputWithoutOutput>(DispatchKey::CPU));
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
      EXPECT_EQ(0, outputs.size());

      EXPECT_TRUE(called);
      EXPECT_TRUE(called_arg2.has_value());
      EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
      EXPECT_FALSE(called_arg3.has_value());
      EXPECT_TRUE(called_arg4.has_value());
      EXPECT_EQ(*called_arg4, "text");

      called = false;
      outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
      EXPECT_EQ(0, outputs.size());

      EXPECT_TRUE(called);
      EXPECT_FALSE(called_arg2.has_value());
      EXPECT_TRUE(called_arg3.has_value());
      EXPECT_EQ(*called_arg3, 4);
      EXPECT_FALSE(called_arg4.has_value());

    */
}

pub fn kernel_with_opt_input_with_output(
        arg1: Tensor,
        arg2: &Option<Tensor>,
        arg3: Option<i64>,
        arg4: Option<String>) -> Option<Tensor> {
    
    todo!();
        /*
            called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
      return arg2;
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_optional_inputs_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithOutput), &kernelWithOptInputWithOutput>(DispatchKey::CPU));
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));

      EXPECT_TRUE(called);
      EXPECT_TRUE(called_arg2.has_value());
      EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
      EXPECT_FALSE(called_arg3.has_value());
      EXPECT_TRUE(called_arg4.has_value());
      EXPECT_EQ(*called_arg4, "text");

      called = false;
      outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
      EXPECT_EQ(1, outputs.size());
      EXPECT_TRUE(outputs[0].isNone());

      EXPECT_TRUE(called);
      EXPECT_FALSE(called_arg2.has_value());
      EXPECT_TRUE(called_arg3.has_value());
      EXPECT_EQ(*called_arg3, 4);
      EXPECT_FALSE(called_arg4.has_value());

    */
}

pub fn kernel_with_opt_input_with_multiple_outputs(
        arg1: Tensor,
        arg2: &Option<Tensor>,
        arg3: Option<i64>,
        arg4: Option<String>) -> (Option<Tensor>,Option<i64>,Option<String>) {
    
    todo!();
        /*
            return std::make_tuple(arg2, arg3, arg4);
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_with_optional_inputs_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithMultipleOutputs), &kernelWithOptInputWithMultipleOutputs>(DispatchKey::CPU));
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
      EXPECT_EQ(3, outputs.size());
      EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
      EXPECT_TRUE(outputs[1].isNone());
      EXPECT_EQ("text", outputs[2].toString()->string());

      outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
      EXPECT_EQ(3, outputs.size());
      EXPECT_TRUE(outputs[0].isNone());
      EXPECT_EQ(4, outputs[1].toInt());
      EXPECT_TRUE(outputs[2].isNone());
    }

    std::string concatKernel(const Tensor& tensor1, std::string a, const std::string& b, i64 c) {
      return a + b + c10::guts::to_string(c);
    }

    void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
      at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, i64>(*op, dummyTensor(dispatch_key), "1", "2", 3);
      EXPECT_EQ("123", result);

    */
}

pub fn expect_cannot_call_concat_boxed(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      expectThrows<c10::Error>(
        [&] {callOp(*op, dummyTensor(dispatch_key), "1", "2", 3);},
        "Tried to call KernelFunction::callBoxed() on a KernelFunction that can only be called with KernelFunction::call()."
      );
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_when_registered_then_can_be_called_unboxed() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", RegisterOperators::options().kernel<decltype(concatKernel), &concatKernel>(DispatchKey::CPU));
      expectCallsConcatUnboxed(DispatchKey::CPU);

    */
}

pub fn kernel_for_schema_inference(
        arg1: Tensor,
        arg2: i64,
        arg3: &List<Tensor>) -> (i64,Tensor) {
    
    todo!();
        /*
            return {};
        */
}

#[test] fn operator_registration_test_function_based_kernel_given_when_registered_without_specifying_schema_then_infers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<decltype(kernelForSchemaInference), &kernelForSchemaInference>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
      ASSERT_TRUE(op.has_value());

      c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
      EXPECT_FALSE(differences.has_value());

    */
}

lazy_static!{
    /*
    template<class Return, class... Args> struct kernel_func final {
      static Return func(Args...) { return {}; }
    };
    template<class... Args> struct kernel_func<void, Args...> final {
      static void func(Args...) {}
    };
    */
}

#[test] fn operator_registration_test_function_based_kernel_given_mismatched_with_different_num_arguments_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));
        }, "The number of arguments is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
        }, "The number of arguments is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
        }, "The number of arguments is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
        }, "The number of arguments is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_mismatched_with_different_argument_type_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor, i64>::func), &kernel_func<i64, Tensor, i64>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor, i64>::func), &kernel_func<i64, Tensor, i64>::func>(DispatchKey::CPU));
        }, "Type mismatch in argument 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor, i64>::func), &kernel_func<i64, Tensor, i64>::func>(DispatchKey::CPU));
        }, "Type mismatch in argument 1: int vs Tensor"
      );

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_mismatched_with_different_num_returns_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 0 vs 1"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 1 vs 0"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 2 vs 0"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
        }, "The number of returns is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_function_based_kernel_given_mismatched_with_different_return_types_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));
        }, "Type mismatch in return 1: Tensor vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<i64, Tensor>::func), &kernel_func<i64, Tensor>::func>(DispatchKey::CPU));
        }, "Type mismatch in return 1: float vs int"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));
        }, "Type mismatch in return 1: float vs Tensor"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, i64>, Tensor>::func), &kernel_func<std::tuple<Tensor, i64>, Tensor>::func>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, i64>, Tensor>::func), &kernel_func<std::tuple<Tensor, i64>, Tensor>::func>(DispatchKey::CPU));
        }, "Type mismatch in return 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, i64>, Tensor>::func), &kernel_func<std::tuple<Tensor, i64>, Tensor>::func>(DispatchKey::CPU));
        }, "Type mismatch in return 1: int vs Tensor"
      );

    */
}
