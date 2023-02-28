/*!
 | This file tests the legacy function-based API
 | for registering kernels.
 |
 | > namespace { Tensor kernel(Tensor a) {...} }
 | > static auto registry = c10::RegisterOperators()
 | >   .op("func(Tensor a) -> Tensor", &kernel);
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/kernel_function_legacy_test.cpp]

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_when_registered_in_constructor_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_multiple_operators_and_kernels_when_registered_in_one_registrar_then_calls_right() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel)
          .op("_test::error(Tensor dummy, int input) -> int", &errorKernel);
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_multiple_operators_and_kernels_when_registered_in_registrars_then_calls_right() {
    todo!();
    /*
    
      auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
      auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", &errorKernel);
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_when_runs_out_of_scope_then_cannot_be_called_anymore() {
    todo!();
    /*
    
      {
        auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);

        expectCallsIncrement(DispatchKey::CPU);
      }

      // now the registrar is destructed. Assert that the schema is gone.
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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", &kernelWithoutOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_zero_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", &kernelWithZeroOutputs);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_output(Tensor dummy, int a, int b) -> int", &kernelWithIntOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::returning_tensor(Tensor input) -> Tensor", &kernelWithTensorOutput);

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
        input3: &Tensor) -> Vec<Tensor> {
    
    todo!();
        /*
            return {input1, input2, input3};
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", &kernelWithTensorListOutput);

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
        input3: i64) -> Vec<i64> {
    
    todo!();
        /*
            return {input1, input2, input3};
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", &kernelWithIntListOutput);

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

pub fn kernel_with_multiple_outputs(_0: Tensor) -> (Tensor,i64,Vec<Tensor>,Option<i64>,Dict<String,Tensor>) {
    
    todo!();
        /*
            Dict<string, Tensor> dict;
      dict.insert("first", dummyTensor(DispatchKey::CPU));
      dict.insert("second", dummyTensor(DispatchKey::CUDA));
      return std::tuple<Tensor, i64, std::vector<Tensor>, c10::optional<i64>, Dict<string, Tensor>>(
        dummyTensor(DispatchKey::CUDA),
        5,
        {dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)},
        c10::optional<i64>(c10::in_place, 0),
        dict
      );
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
         .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", &kernelWithMultipleOutputs);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_input_by_reference_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", &kernelWithTensorInputByReferenceWithOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_input_by_value_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", &kernelWithTensorInputByValueWithOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_input_by_reference_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", &kernelWithTensorInputByReferenceWithoutOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_input_by_value_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", &kernelWithTensorInputByValueWithoutOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> ()", &kernelWithIntInputWithoutOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> int", &kernelWithIntInputWithOutput);

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
        input1: &Vec<i64>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", &kernelWithIntListInputWithoutOutput);

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
        input1: &Vec<i64>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_int_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> int", &kernelWithIntListInputWithOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(3, outputs[0].toInt());

    */
}

pub fn kernel_with_tensor_list_input_without_output(input1: &Vec<Tensor>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithTensorListInputWithoutOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

pub fn kernel_with_tensor_list_input_with_output(input1: &Vec<Tensor>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithTensorListInputWithOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(2, outputs[0].toInt());

    */
}

pub fn kernel_with_legacy_tensor_vector_input_without_output(input1: &Vec<Tensor>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_vector_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithLegacyTensorVectorInputWithoutOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

pub fn kernel_with_legacy_tensor_vector_input_with_output(input1: &Vec<Tensor>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_vector_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithLegacyTensorVectorInputWithOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(2, outputs[0].toInt());

    */
}


pub fn kernel_with_legacy_tensor_list_input_without_output(input1: Vec<Tensor>)  {
    
    todo!();
        /*
            captured_input_list_size = input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithLegacyTensorListInputWithoutOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

pub fn kernel_with_legacy_tensor_list_input_with_output(input1: Vec<Tensor>) -> i64 {
    
    todo!();
        /*
            return input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_tensor_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithLegacyTensorListInputWithOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(2, outputs[0].toInt());

    */
}

pub fn kernel_with_string_list_output(input: Vec<String>) -> Vec<String> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_string_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::stringlist_output(str[] input) -> str[]", &kernelWithStringListOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::stringlist_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::List<std::string> list({"value1", "value2"});
      auto outputs = callOp(*op, list);
      EXPECT_EQ(1, outputs.size());
      auto output = std::move(outputs[0]).toList();

      EXPECT_EQ(2, output.size());
      EXPECT_EQ("value1", output.get(0).toString()->string());
      EXPECT_EQ("value2", output.get(1).toString()->string());

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_dict_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, Tensor) input) -> ()", &kernelWithDictInputWithoutOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_dict_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, str) input) -> str", &kernelWithDictInputWithOutput);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_dict_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", &kernelWithDictOutput);

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

pub fn kernel_with_unordered_map_input_without_output(input1: HashMap<String,Tensor>)  {
    
    todo!();
        /*
            captured_dict_size = input1.size();
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_unordered_map_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, Tensor) input) -> ()", &kernelWithUnorderedMapInputWithoutOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_dict_size = 0;
      c10::Dict<string, Tensor> dict;
      dict.insert("key1", dummyTensor(DispatchKey::CPU));
      dict.insert("key2", dummyTensor(DispatchKey::CUDA));
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_dict_size);

    */
}

pub fn kernel_with_unordered_map_input_with_output(input1: HashMap<String,String>) -> String {
    
    todo!();
        /*
            return input1.at("key2");
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_unordered_map_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, str) input) -> str", &kernelWithUnorderedMapInputWithOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, string> dict;
      dict.insert("key1", "value1");
      dict.insert("key2", "value2");
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ("value2", outputs[0].toString()->string());

    */
}

pub fn kernel_with_unordered_map_output(input: HashMap<String,String>) -> HashMap<String,String> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_unordered_map_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", &kernelWithUnorderedMapOutput);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, string> dict;
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


pub fn kernel_with_map_of_int_list(input: HashMap<String,Vec<i64>>) -> HashMap<String,Vec<i64>> {
    
    todo!();
        /*
            return input;
        */
}



#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_map_of_list_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, int[]) input) -> Dict(str, int[])", &kernelWithMapOfIntList);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, c10::List<i64>> dict;
      dict.insert("key1", c10::List<i64>({10, 20}));
      dict.insert("key2", c10::List<i64>({30, 40}));
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(1, outputs.size());
      auto output = c10::impl::toTypedDict<string, c10::List<i64>>(outputs[0].toGenericDict());

      EXPECT_EQ(2, output.size());
      EXPECT_EQ(2, output.at("key1").size());
      EXPECT_EQ(10, output.at("key1").get(0));
      EXPECT_EQ(20, output.at("key1").get(1));
      EXPECT_EQ(2, output.at("key2").size());
      EXPECT_EQ(30, output.at("key2").get(0));
      EXPECT_EQ(40, output.at("key2").get(1));

    */
}

pub fn kernel_with_map_of_list_of_map(input: HashMap<String,Vec<HashMap<i64,String>>>) -> HashMap<String,Vec<HashMap<i64,String>>> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_map_of_list_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, Dict(int,str)[]) input) -> Dict(str, Dict(int,str)[])", &kernelWithMapOfListOfMap);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, c10::List<c10::Dict<i64, string>>> dict;
      c10::Dict<i64, string> dict1;
      dict1.insert(10, "10");
      dict1.insert(20, "20");
      dict.insert("key1", c10::List<c10::Dict<i64, string>>({dict1}));
      c10::Dict<i64, string> dict2;
      dict2.insert(30, "30");
      dict2.insert(40, "40");
      dict.insert("key2", c10::List<c10::Dict<i64, string>>({dict2}));
      auto outputs = callOp(*op, dict);
      EXPECT_EQ(1, outputs.size());
      auto output = c10::impl::toTypedDict<string, c10::List<c10::Dict<i64, string>>>(outputs[0].toGenericDict());

      EXPECT_EQ(2, output.size());
      EXPECT_EQ(1, output.at("key1").size());
      EXPECT_EQ(2, output.at("key1").get(0).size());
      EXPECT_EQ("10", output.at("key1").get(0).at(10));
      EXPECT_EQ("20", output.at("key1").get(0).at(20));
      EXPECT_EQ(2, output.at("key2").get(0).size());
      EXPECT_EQ("30", output.at("key2").get(0).at(30));
      EXPECT_EQ("40", output.at("key2").get(0).at(40));

    */
}

pub fn kernel_with_list_of_map(input: Vec<HashMap<String,i64>>) -> Vec<HashMap<String,i64>> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_list_of_map_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Dict(str, int)[] input) -> Dict(str, int)[]", &kernelWithListOfMap);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, i64> dict1;
      dict1.insert("1", 1);
      dict1.insert("2", 2);
      c10::Dict<string, i64> dict2;
      dict2.insert("3", 3);
      dict2.insert("4", 4);
      c10::List<c10::Dict<string, i64>> list({dict1, dict2});
      auto outputs = callOp(*op, list);
      EXPECT_EQ(1, outputs.size());
      c10::impl::GenericList output = std::move(outputs[0]).toList();

      EXPECT_EQ(2, output.size());
      EXPECT_EQ(2, output.get(0).toGenericDict().size());
      EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toInt());
      EXPECT_EQ(2, output.get(0).toGenericDict().at("2").toInt());
      EXPECT_EQ(2, output.get(1).toGenericDict().size());
      EXPECT_EQ(3, output.get(1).toGenericDict().at("3").toInt());
      EXPECT_EQ(4, output.get(1).toGenericDict().at("4").toInt());

    */
}

pub fn kernel_with_list_of_map_of_int_list(input: Vec<HashMap<String,Vec<i64>>>) -> Vec<HashMap<String,Vec<i64>>> {
    
    todo!();
        /*
            return input;
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_list_of_map_int_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Dict(str, int[])[] input) -> Dict(str, int[])[]", &kernelWithListOfMapOfIntList);

      auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
      ASSERT_TRUE(op.has_value());

      c10::Dict<string, c10::List<i64>> dict1;
      dict1.insert("1", c10::List<i64>({1, 2}));
      dict1.insert("3", c10::List<i64>({3, 4}));
      c10::Dict<string, c10::List<i64>> dict2;
      dict2.insert("5", c10::List<i64>({5, 6}));
      dict2.insert("7", c10::List<i64>({7, 8}));
      c10::List<c10::Dict<string, c10::List<i64>>> list({ dict1, dict2 });
      auto outputs = callOp(*op, list);
      EXPECT_EQ(1, outputs.size());
      c10::impl::GenericList output = std::move(outputs[0]).toList();

      EXPECT_EQ(2, output.size());
      EXPECT_EQ(2, output.get(0).toGenericDict().size());
      EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector().size());
      EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toIntVector()[0]);
      EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector()[1]);
      EXPECT_EQ(2, output.get(0).toGenericDict().at("3").toIntVector().size());
      EXPECT_EQ(3, output.get(0).toGenericDict().at("3").toIntVector()[0]);
      EXPECT_EQ(4, output.get(0).toGenericDict().at("3").toIntVector()[1]);
      EXPECT_EQ(2, output.get(1).toGenericDict().at("5").toIntVector().size());
      EXPECT_EQ(5, output.get(1).toGenericDict().at("5").toIntVector()[0]);
      EXPECT_EQ(6, output.get(1).toGenericDict().at("5").toIntVector()[1]);
      EXPECT_EQ(2, output.get(1).toGenericDict().at("7").toIntVector().size());
      EXPECT_EQ(7, output.get(1).toGenericDict().at("7").toIntVector()[0]);
      EXPECT_EQ(8, output.get(1).toGenericDict().at("7").toIntVector()[1]);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_fallback_without_any_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args() -> ()", &kernelWithoutInputs);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_fallback_without_tensor_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args(int arg) -> int", &kernelWithoutTensorInputs);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_optional_inputs_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", &kernelWithOptInputWithoutOutput);
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
      EXPECT_EQ(0, outputs.size());

      EXPECT_TRUE(called);
      EXPECT_TRUE(called_arg2.has_value());
      EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_optional_inputs_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", &kernelWithOptInputWithOutput);
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));

      EXPECT_TRUE(called);
      EXPECT_TRUE(called_arg2.has_value());
      EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_with_optional_inputs_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", &kernelWithOptInputWithMultipleOutputs);
      auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
      EXPECT_EQ(3, outputs.size());
      EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));
      EXPECT_TRUE(outputs[1].isNone());
      EXPECT_EQ("text", outputs[2].toString()->string());

      outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
      EXPECT_EQ(3, outputs.size());
      EXPECT_TRUE(outputs[0].isNone());
      EXPECT_EQ(4, outputs[1].toInt());
      EXPECT_TRUE(outputs[2].isNone());

    */
}

pub fn concat_kernel(
        tensor1: &Tensor,
        a:       String,
        b:       &String,
        c:       i64) -> String {
    
    todo!();
        /*
            return a + b + c10::guts::to_string(c);
        */
}

pub fn expect_calls_concat_unboxed(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, i64>(*op, dummyTensor(dispatch_key), "1", "2", 3);
      EXPECT_EQ("123", result);
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_when_registered_then_can_be_called_unboxed() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", &concatKernel);
      expectCallsConcatUnboxed(DispatchKey::CPU);

    */
}

pub fn kernel_for_schema_inference(
        arg1: Tensor,
        arg2: i64,
        arg3: &Vec<Tensor>) -> (i64,Tensor) {
    
    todo!();
        /*
            return {};
        */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_when_registered_without_specifying_schema_then_infers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::no_schema_specified", &kernelForSchemaInference);

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

#[test] fn operator_registration_test_legacy_function_based_kernel_given_mismatched_with_different_num_arguments_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", &kernel_func<i64, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", &kernel_func<i64, Tensor>::func);
        }, "The number of arguments is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", &kernel_func<void, Tensor, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch() -> ()", &kernel_func<void, Tensor, Tensor>::func);
        }, "The number of arguments is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<void, Tensor, Tensor>::func);
        }, "The number of arguments is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", &kernel_func<void, Tensor, Tensor>::func);
        }, "The number of arguments is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_mismatched_with_different_argument_type_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg1, int arg2) -> int", &kernel_func<i64, Tensor, i64>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg1, float arg2) -> int", &kernel_func<i64, Tensor, i64>::func);
        }, "Type mismatch in argument 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(int arg1, int arg2) -> int", &kernel_func<i64, Tensor, i64>::func);
        }, "Type mismatch in argument 1: int vs Tensor"
      );

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_mismatched_with_different_num_returns_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", &kernel_func<i64, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<i64, Tensor>::func);
        }, "The number of returns is different. 0 vs 1"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", &kernel_func<i64, Tensor>::func);
        }, "The number of returns is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<void, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<void, Tensor>::func);
        }, "The number of returns is different. 1 vs 0"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", &kernel_func<void, Tensor>::func);
        }, "The number of returns is different. 2 vs 0"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
        }, "The number of returns is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
        }, "The number of returns is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
        }, "The number of returns is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_legacy_function_based_kernel_given_mismatched_with_different_return_types_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", &kernel_func<i64, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<i64, Tensor>::func);
        }, "Type mismatch in return 1: Tensor vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", &kernel_func<i64, Tensor>::func);
        }, "Type mismatch in return 1: float vs int"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<Tensor, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", &kernel_func<Tensor, Tensor>::func);
        }, "Type mismatch in return 1: float vs Tensor"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, int)", &kernel_func<std::tuple<Tensor, i64>, Tensor>::func);

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, float)", &kernel_func<std::tuple<Tensor, i64>, Tensor>::func);
        }, "Type mismatch in return 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", &kernel_func<std::tuple<Tensor, i64>, Tensor>::func);
        }, "Type mismatch in return 1: int vs Tensor"
      );

    */
}
