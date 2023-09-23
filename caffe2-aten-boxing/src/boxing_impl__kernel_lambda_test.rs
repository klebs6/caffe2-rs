crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/kernel_lambda_test.cpp]

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

#[test] fn operator_registration_test_lambda_based_kernel_given_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 i) {return i+1;}));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_out_of_line_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto my_kernel = [] (Tensor, i64 i) {return i+1;};
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, my_kernel));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_multiple_operators_and_kernels_when_registered_in_one_registrar_then_calls_right() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 i) {return i+1;})
                                                                                          .kernel(DispatchKey::CUDA, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;}))
          .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;})
                                                                                          .kernel(DispatchKey::CUDA, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;}));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_multiple_operators_and_kernels_when_registered_in_registrars_then_calls_right() {
    todo!();
    /*
    
      auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 i) {return i+1;})
                                                                                                                           .kernel(DispatchKey::CUDA, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;}));
      auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;})
                                                                                                                           .kernel(DispatchKey::CUDA, [] (Tensor, i64) -> i64 {EXPECT_TRUE(false); return 0;}));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_when_runs_out_of_scope_then_cannot_be_called_anymore() {
    todo!();
    /*
    
      {
        auto m = MAKE_TORCH_LIBRARY(_test);
        m.def("_test::my_op(Tensor dummy, int input) -> int");
        auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
        m_cpu.impl("my_op", DispatchKey::CPU, [] (Tensor, i64 i) {return i+1;});
        {
          auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
          m_cuda.impl("my_op", DispatchKey::CUDA, [] (Tensor, i64 i) {return i-1;});

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

#[test] fn operator_registration_test_lambda_based_kernel_given_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()",
        RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (const Tensor&) -> void {was_called = true;}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_zero_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&) -> std::tuple<> {was_called = true; return {};}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_output(Tensor dummy, int a, int b) -> int",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 a, i64 b) {return a+b;}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(9, result[0].toInt());

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::returning_tensor(Tensor input) -> Tensor",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                        .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]",
            RegisterOperators::options().kernel(DispatchKey::CUDA, [] (const Tensor& a, const Tensor& b, const Tensor& c) -> c10::List<Tensor> {return c10::List<Tensor>({a, b, c});}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&, i64 a, i64 b, i64 c) -> c10::List<i64> {return c10::List<i64>({a,b,c});}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
         .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))",
           RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, i64, c10::List<Tensor>, c10::optional<i64>, Dict<string, Tensor>> {
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
           }));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_input_by_reference_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                        .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_input_by_value_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor a) {return a;})
                                        .kernel(DispatchKey::CUDA, [] (Tensor a) {return a;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_input_by_reference_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) -> void {captured_input = a;})
                                        .kernel(DispatchKey::CUDA, [] (const Tensor& a) -> void {captured_input = a;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_input_by_value_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor a) -> void {captured_input = a;})
                                        .kernel(DispatchKey::CUDA, [] (Tensor a) -> void {captured_input = a;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> ()",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 a) -> void {captured_int_input = a;}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_int_input = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_int_input);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> int",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64 a) {return a + 1;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> ()",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, const c10::List<i64>& a) {captured_input_list_size = a.size();}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_input_list_size);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_int_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> int",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, const c10::List<i64>& a) -> i64 {return a.size();}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(3, outputs[0].toInt());

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()",
            RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> void {captured_input_list_size = a.size();}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_with_tensor_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int",
             RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> i64 {return a.size();}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_dict_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel([] (Dict<string, Tensor> input1) {
            captured_dict_size = input1.size();
          }));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_dict_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input1) {
            return input1.at("key2");
          }));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_dict_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
        .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input) {
          return input;
        }));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_fallback_without_any_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel([] () {called = true;}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op);
      EXPECT_TRUE(called);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_fallback_without_tensor_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel([] (i64 arg) {return arg + 1;}));

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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_optional_inputs_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op(
        "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<i64> arg3, c10::optional<std::string> arg4) {
          called = true;
          called_arg2 = arg2;
          called_arg3 = arg3;
          called_arg4 = arg4;
        }));
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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_optional_inputs_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op(
        "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<i64> arg3, c10::optional<std::string> arg4) {
          called = true;
          called_arg2 = arg2;
          called_arg3 = arg3;
          called_arg4 = arg4;
          return arg2;
        }));
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

#[test] fn operator_registration_test_lambda_based_kernel_given_with_optional_inputs_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op(
        "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const c10::optional<Tensor>& arg2, c10::optional<i64> arg3, c10::optional<std::string> arg4) {
          return std::make_tuple(arg2, arg3, arg4);
        }));
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

#[test] fn operator_registration_test_lambda_based_kernel_given_when_registered_then_can_be_called_unboxed() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", torch::RegisterOperators::options()
        .kernel(DispatchKey::CPU, [] (const Tensor& tensor1, std::string a, const std::string& b, i64 c) {
          return a + b + c10::guts::to_string(c);
        }));
      expectCallsConcatUnboxed(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_when_registered_without_specifying_schema_then_infers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel([] (Tensor arg1, i64 arg2, const c10::List<Tensor>& arg3) -> std::tuple<i64, Tensor> {return {};}));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
      ASSERT_TRUE(op.has_value());

      c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
      EXPECT_FALSE(differences.has_value());

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_mismatched_with_different_num_arguments_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));
        }, "The number of arguments is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch() -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
        }, "The number of arguments is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
        }, "The number of arguments is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
        }, "The number of arguments is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_mismatched_with_different_argument_type_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64) -> i64 {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64) -> i64 {return {};}));
        }, "Type mismatch in argument 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, i64) -> i64 {return {};}));
        }, "Type mismatch in argument 1: int vs Tensor"
      );

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_mismatched_with_different_num_returns_when_registering_then_fails() {

    todo!();

    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));
        }, "The number of returns is different. 0 vs 1"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));
        }, "The number of returns is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
        }, "The number of returns is different. 1 vs 0"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
        }, "The number of returns is different. 2 vs 0"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
        }, "The number of returns is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
        }, "The number of returns is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
        }, "The number of returns is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_lambda_based_kernel_given_mismatched_with_different_return_types_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));
        }, "Type mismatch in return 1: Tensor vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> i64 {return {};}));
        }, "Type mismatch in return 1: float vs int"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));
        }, "Type mismatch in return 1: float vs Tensor"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, i64> {return {};}));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, i64> {return {};}));
        }, "Type mismatch in return 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, i64> {return {};}));
        }, "Type mismatch in return 1: int vs Tensor"
      );

    */
}
