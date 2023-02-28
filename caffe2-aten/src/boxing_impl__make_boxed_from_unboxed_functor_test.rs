crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor_test.cpp]

pub struct ErrorKernel {
    base: OperatorKernel,
}

impl ErrorKernel {
    
    pub fn invoke(&mut self, 
        _0: &Tensor,
        _1: i64) -> i64 {
        
        todo!();
        /*
            EXPECT_TRUE(false); // this kernel should never be called
        return 0;
        */
    }
}

//------------------
pub struct IncrementKernel {
    base: OperatorKernel,
}

impl IncrementKernel {

    pub fn invoke(&mut self, 
        tensor: &Tensor,
        input:  i64) -> i64 {
        
        todo!();
        /*
            return input + 1;
        */
    }
}

//------------------
pub struct DecrementKernel {
    base: OperatorKernel,
}

impl DecrementKernel {
    
    pub fn invoke(&mut self, 
        tensor: &Tensor,
        input:  i64) -> i64 {
        
        todo!();
        /*
            return input - 1;
        */
    }
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

#[test] fn operator_registration_test_functor_based_kernel_given_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(DispatchKey::CPU));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_multiple_operators_and_kernels_when_registered_in_one_registrar_then_calls_right() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(DispatchKey::CPU)
                                                                                          .kernel<ErrorKernel>(DispatchKey::CUDA))
          .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(DispatchKey::CPU)
                                                                                          .kernel<ErrorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_multiple_operators_and_kernels_when_registered_in_registrars_then_calls_right() {
    todo!();
    /*
    
      auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<ErrorKernel>(DispatchKey::CUDA));
      auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(DispatchKey::CPU)
                                                                                                                           .kernel<ErrorKernel>(DispatchKey::CUDA));
      expectCallsIncrement(DispatchKey::CPU);

    */
}

lazy_static!{
    /*
    bool was_called = false;
    */
}

//-------------------
pub struct KernelWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithoutOutput {
    
    pub fn invoke(&mut self, _0: &Tensor)  {
        
        todo!();
        /*
            was_called = true;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", RegisterOperators::options().kernel<KernelWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

//-----------------------------
pub struct KernelWithZeroOutputs {
    base: OperatorKernel,
}

impl KernelWithZeroOutputs {

    pub fn invoke(&mut self, _0: &Tensor) -> () {
        
        todo!();
        /*
            was_called = true;
        return std::make_tuple();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_zero_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", RegisterOperators::options().kernel<KernelWithZeroOutputs>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
      ASSERT_TRUE(op.has_value());
      was_called = false;
      auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
      EXPECT_TRUE(was_called);
      EXPECT_EQ(0, result.size());

    */
}

pub struct KernelWithIntOutput {
    base: OperatorKernel,
}

impl KernelWithIntOutput {
    
    pub fn invoke(&mut self, 
        _0: Tensor,
        a:  i64,
        b:  i64) -> i64 {
        
        todo!();
        /*
            return a + b;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_output(Tensor dummy, int a, int b) -> int", RegisterOperators::options().kernel<KernelWithIntOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
      ASSERT_TRUE(op.has_value());

      auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
      EXPECT_EQ(1, result.size());
      EXPECT_EQ(9, result[0].toInt());

    */
}

//---------------------------
pub struct KernelWithTensorOutput {
    base: OperatorKernel,
}

impl KernelWithTensorOutput {
    
    pub fn invoke(&mut self, input: &Tensor) -> Tensor {
        
        todo!();
        /*
            return input;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorOutput>(DispatchKey::CPU)
                                                                                             .kernel<KernelWithTensorOutput>(DispatchKey::CUDA));

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

//---------------------------
pub struct KernelWithTensorListOutput {
    base: OperatorKernel,
}

impl KernelWithTensorListOutput {
    
    pub fn invoke(&mut self, 
        input1: &Tensor,
        input2: &Tensor,
        input3: &Tensor) -> List<Tensor> {
        
        todo!();
        /*
            return c10::List<Tensor>({input1, input2, input3});
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<KernelWithTensorListOutput>(DispatchKey::CUDA));

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

//------------------------------
pub struct KernelWithIntListOutput {
    base: OperatorKernel,
}

impl KernelWithIntListOutput {
    
    pub fn invoke(&mut self, 
        _0:     &Tensor,
        input1: i64,
        input2: i64,
        input3: i64) -> List<i64> {
        
        todo!();
        /*
            return c10::List<i64>({input1, input2, input3});
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_list_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<KernelWithIntListOutput>(DispatchKey::CPU));

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

//---------------------------
pub struct KernelWithMultipleOutputs {
    base: OperatorKernel,
}

impl KernelWithMultipleOutputs {
    
    pub fn invoke(&mut self, _0: Tensor) -> (Tensor,i64,List<Tensor>,Option<i64>,Dict<String,Tensor>) {
        
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
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
         .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<KernelWithMultipleOutputs>(DispatchKey::CPU));

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

//-------------------------
pub struct KernelWithTensorInputByReferenceWithOutput {
    base: OperatorKernel,
}

impl KernelWithTensorInputByReferenceWithOutput {
    
    pub fn invoke(&mut self, input1: &Tensor) -> Tensor {
        
        todo!();
        /*
            return input1;
        */
    }
}

//-------------------------
pub struct KernelWithTensorInputByValueWithOutput {
    base: OperatorKernel,
}

impl KernelWithTensorInputByValueWithOutput {

    pub fn invoke(&mut self, input1: Tensor) -> Tensor {
        
        todo!();
        /*
            return input1;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_input_by_reference_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithOutput>(DispatchKey::CPU)
                                                                                         .kernel<KernelWithTensorInputByReferenceWithOutput>(DispatchKey::CUDA));

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

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_input_by_value_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithOutput>(DispatchKey::CPU)
                                                                                         .kernel<KernelWithTensorInputByValueWithOutput>(DispatchKey::CUDA));

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

//-----------------------------
pub struct KernelWithTensorInputByReferenceWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithTensorInputByReferenceWithoutOutput {

    pub fn invoke(&mut self, input1: &Tensor)  {
        
        todo!();
        /*
            captured_input = input1;
        */
    }
}

//-----------------------------
pub struct KernelWithTensorInputByValueWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithTensorInputByValueWithoutOutput {
    
    pub fn invoke(&mut self, input1: Tensor)  {
        
        todo!();
        /*
            captured_input = input1;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_input_by_reference_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CPU)
                                                                                     .kernel<KernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CUDA));

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

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_input_by_value_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithoutOutput>(DispatchKey::CPU)
                                                                                     .kernel<KernelWithTensorInputByValueWithoutOutput>(DispatchKey::CUDA));

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

pub struct KernelWithIntInputWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithIntInputWithoutOutput {
    
    pub fn invoke(&mut self, 
        _0:     Tensor,
        input1: i64)  {
        
        todo!();
        /*
            captured_int_input = input1;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options().kernel<KernelWithIntInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_int_input = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_int_input);

    */
}

pub struct KernelWithIntInputWithOutput {
    base: OperatorKernel,
}

impl KernelWithIntInputWithOutput {
    
    pub fn invoke(&mut self, 
        _0:     Tensor,
        input1: i64) -> i64 {
        
        todo!();
        /*
            return input1 + 1;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<KernelWithIntInputWithOutput>(DispatchKey::CPU));

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

pub struct KernelWithIntListInputWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithIntListInputWithoutOutput {
    
    pub fn invoke(&mut self, 
        _0:     Tensor,
        input1: &List<i64>)  {
        
        todo!();
        /*
            captured_input_list_size = input1.size();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<KernelWithIntListInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(3, captured_input_list_size);

    */
}

pub struct KernelWithIntListInputWithOutput {
    base: OperatorKernel,
}

impl KernelWithIntListInputWithOutput {
    
    pub fn invoke(&mut self, 
        _0:     Tensor,
        input1: &List<i64>) -> i64 {
        
        todo!();
        /*
            return input1.size();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_int_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<KernelWithIntListInputWithOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<i64>({2, 4, 6}));
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(3, outputs[0].toInt());

    */
}

//----------------------------
pub struct KernelWithTensorListInputWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithTensorListInputWithoutOutput {
    
    pub fn invoke(&mut self, input1: &List<Tensor>)  {
        
        todo!();
        /*
            captured_input_list_size = input1.size();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_list_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<KernelWithTensorListInputWithoutOutput>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
      ASSERT_TRUE(op.has_value());

      captured_input_list_size = 0;
      auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
      EXPECT_EQ(0, outputs.size());
      EXPECT_EQ(2, captured_input_list_size);

    */
}

//--------------------------------
pub struct KernelWithTensorListInputWithOutput {
    base: OperatorKernel,
}

impl KernelWithTensorListInputWithOutput {
    
    pub fn invoke(&mut self, input1: &List<Tensor>) -> i64 {
        
        todo!();
        /*
            return input1.size();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tensor_list_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<KernelWithTensorListInputWithOutput>(DispatchKey::CPU));

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

pub struct KernelWithDictInputWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithDictInputWithoutOutput {
    
    pub fn invoke(&mut self, input1: Dict<String,Tensor>)  {
        
        todo!();
        /*
            captured_dict_size = input1.size();
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_dict_input_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithoutOutput>());

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

pub struct KernelWithDictInputWithOutput {
    base: OperatorKernel,
}

impl KernelWithDictInputWithOutput {

    pub fn invoke(&mut self, input1: Dict<String,String>) -> String {
        
        todo!();
        /*
            return input1.at("key2");
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_dict_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithOutput>());

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

pub struct KernelWithDictOutput {
    base: OperatorKernel,
}

impl KernelWithDictOutput {
    
    pub fn invoke(&mut self, input: Dict<String,String>) -> Dict<String,String> {
        
        todo!();
        /*
            return input;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_dict_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<KernelWithDictOutput>());

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

//--------------------------------
pub struct KernelWithCache {
    base:    OperatorKernel,
    counter: i64,
}

impl Default for KernelWithCache {
    
    fn default() -> Self {
        todo!();
        /*
        : counter(3),

        
        */
    }
}

impl KernelWithCache {
    
    pub fn invoke(&mut self, _0: Tensor) -> i64 {
        
        todo!();
        /*
            return ++counter;
        */
    }
}

//----------------------------
pub struct KernelWithTupleInput {
    base: OperatorKernel,
}

impl KernelWithTupleInput {
    
    pub fn invoke(&mut self, input1: (String,i64,f64)) -> String {
        
        todo!();
        /*
            return std::get<0>(input1);
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_tuple_input_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::tuple_input((str, int, float) input) -> str", RegisterOperators::options().catchAllKernel<KernelWithTupleInput>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::tuple_input", ""});
      ASSERT_TRUE(op.has_value());

      std::tuple<string, i64, float> tup{"foobar", 123, 420.1337};
      auto outputs = callOp(*op, tup);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ("foobar", outputs[0].toString()->string());

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_cache_then_is_kept_correctly() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::cache_op(Tensor input) -> int", RegisterOperators::options().kernel<KernelWithCache>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::cache_op", ""});
      ASSERT_TRUE(op.has_value());

      // expect first time calling returns a 4 (4 is the initial value in the cache)
      auto stack = makeStack(dummyTensor(DispatchKey::CPU));
      op->callBoxed(&stack);
      EXPECT_EQ(1, stack.size());
      EXPECT_EQ(4, stack[0].toInt());

      // expect second time calling returns a 5
      stack = makeStack(dummyTensor(DispatchKey::CPU));
      op->callBoxed(&stack);
      EXPECT_EQ(1, stack.size());
      EXPECT_EQ(5, stack[0].toInt());

      // expect third time calling returns a 6
      stack = makeStack(dummyTensor(DispatchKey::CPU));
      op->callBoxed(&stack);
      EXPECT_EQ(1, stack.size());
      EXPECT_EQ(6, stack[0].toInt());

    */
}

//----------------------------
pub struct KernelWithConstructorArg {
    base:    OperatorKernel,
    offset: i64,
}

impl KernelWithConstructorArg {
    
    pub fn new(offset: i64) -> Self {
    
        todo!();
        /*
        : offset(offset),

        
        */
    }
    
    pub fn invoke(&mut self, 
        _0:    &Tensor,
        input: i64) -> i64 {
        
        todo!();
        /*
            return input + offset_;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_constructor_arg_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithConstructorArg>(DispatchKey::CPU, 2)
                                                                                               .kernel<KernelWithConstructorArg>(DispatchKey::CUDA, 4));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 4);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(6, outputs[0].toInt());

      outputs = callOp(*op, dummyTensor(DispatchKey::CUDA), 4);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(8, outputs[0].toInt());

    */
}

//------------------------------------
pub struct KernelWithMultipleConstructorArgs {
    base:   OperatorKernel,
    offset: i64,
}

impl KernelWithMultipleConstructorArgs {
    
    pub fn new(
        offset1: i64,
        offset2: i64) -> Self {
    
        todo!();
        /*


            : offset_(offset1 + offset2)
        */
    }
    
    pub fn invoke(&mut self, 
        _0:    &Tensor,
        input: i64) -> i64 {
        
        todo!();
        /*
            return input + offset_;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_multiple_constructor_args_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithMultipleConstructorArgs>(DispatchKey::CPU, 2, 3)
                                                                                               .kernel<KernelWithMultipleConstructorArgs>(DispatchKey::CUDA, 4, 5));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
      ASSERT_TRUE(op.has_value());

      auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 4);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(9, outputs[0].toInt());

      outputs = callOp(*op, dummyTensor(DispatchKey::CUDA), 4);
      EXPECT_EQ(1, outputs.size());
      EXPECT_EQ(13, outputs[0].toInt());

    */
}

lazy_static!{
    /*
    bool called = false;
    */
}

pub struct KernelWithoutInputs {
    base: OperatorKernel,
}

impl KernelWithoutInputs {
    
    pub fn invoke(&mut self)  {
        
        todo!();
        /*
            called = true;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_fallback_without_any_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<KernelWithoutInputs>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
      ASSERT_TRUE(op.has_value());

      called = false;
      auto outputs = callOp(*op);
      EXPECT_TRUE(called);

    */
}

pub struct KernelWithoutTensorInputs {
    base: OperatorKernel,
}

impl KernelWithoutTensorInputs {

    pub fn invoke(&mut self, arg: i64) -> i64 {
        
        todo!();
        /*
            return arg + 1;
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_fallback_without_tensor_arguments_when_registered_then_can_be_called() {
    todo!();
    /*
    
      // note: non-fallback kernels without tensor arguments don't work because there
      // is no way to get the dispatch key. For operators that only have a fallback
      // kernel, this must work for backwards compatibility.
      auto registrar = RegisterOperators()
          .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<KernelWithoutTensorInputs>());

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

pub struct KernelWithOptInputWithoutOutput {
    base: OperatorKernel,
}

impl KernelWithOptInputWithoutOutput {
    
    pub fn invoke(&mut self, 
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
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_optional_inputs_without_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<KernelWithOptInputWithoutOutput>(DispatchKey::CPU));
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

//----------------------------
pub struct KernelWithOptInputWithOutput {
    base: OperatorKernel,
}

impl KernelWithOptInputWithOutput {
    
    pub fn invoke(&mut self, 
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
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_optional_inputs_output_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<KernelWithOptInputWithOutput>(DispatchKey::CPU));
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

pub struct KernelWithOptInputWithMultipleOutputs {
    base: OperatorKernel,
}

impl KernelWithOptInputWithMultipleOutputs {

    pub fn invoke(&mut self, 
        arg1: Tensor,
        arg2: &Option<Tensor>,
        arg3: Option<i64>,
        arg4: Option<String>) -> (Option<Tensor>,Option<i64>,Option<String>) {
        
        todo!();
        /*
            return std::make_tuple(arg2, arg3, arg4);
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_with_optional_inputs_multiple_outputs_when_registered_then_can_be_called() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<KernelWithOptInputWithMultipleOutputs>(DispatchKey::CPU));
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

//---------------------------------
pub struct ConcatKernel {
    base:   OperatorKernel,
    prefix: String,
}

impl ConcatKernel {

    pub fn new(prefix: String) -> Self {
    
        todo!();
        /*
        : prefix(std::move(prefix)),

        
        */
    }
    
    pub fn invoke(&mut self, 
        tensor1: &Tensor,
        a:       String,
        b:       &String,
        c:       i64) -> String {
        
        todo!();
        /*
            return prefix_ + a + b + c10::guts::to_string(c);
        */
    }
}

pub fn expect_calls_concat_unboxed(dispatch_key: DispatchKey)  {
    
    todo!();
        /*
            at::AutoDispatchBelowAutograd mode;

      // assert that schema and cpu kernel are present
      auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
      ASSERT_TRUE(op.has_value());
      std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, i64>(*op, dummyTensor(dispatch_key), "1", "2", 3);
      EXPECT_EQ("prefix123", result);
        */
}

#[test] fn operator_registration_test_functor_based_kernel_given_when_registered_then_can_be_called_unboxed() {
    todo!();
    /*
    
      auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", RegisterOperators::options().kernel<ConcatKernel>(DispatchKey::CPU, "prefix"));
      expectCallsConcatUnboxed(DispatchKey::CPU);

    */
}

//---------------------------
pub struct KernelForSchemaInference {
    base: OperatorKernel,
}

impl KernelForSchemaInference {
    
    pub fn invoke(&mut self, 
        arg1: Tensor,
        arg2: i64,
        arg3: &List<Tensor>) -> (i64,Tensor) {
        
        todo!();
        /*
            return {};
        */
    }
}

#[test] fn operator_registration_test_functor_based_kernel_given_when_registered_without_specifying_schema_then_infers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::no_schema_specified", RegisterOperators::options().kernel<KernelForSchemaInference>(DispatchKey::CPU));

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
      ASSERT_TRUE(op.has_value());

      c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
      EXPECT_FALSE(differences.has_value());

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_when_registered_catch_all_without_specifying_schema_then_infers() {
    todo!();
    /*
    
      auto registrar = RegisterOperators()
          .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<KernelForSchemaInference>());

      auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
      ASSERT_TRUE(op.has_value());

      c10::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
      EXPECT_FALSE(differences.has_value());

    */
}

lazy_static!{
    /*
    template<class Return, class... Args> struct KernelFunc final : OperatorKernel{
      Return operator()(Args...) { return {}; }
    };
    template<class... Args> struct KernelFunc<void, Args...> final : OperatorKernel {
      void operator()(Args...) {}
    };
    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_mismatched_with_different_num_arguments_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));
        }, "The number of arguments is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
        }, "The number of arguments is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
        }, "The number of arguments is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
        }, "The number of arguments is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_mismatched_with_different_argument_type_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor, i64>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor, i64>>(DispatchKey::CPU));
        }, "Type mismatch in argument 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor, i64>>(DispatchKey::CPU));
        }, "Type mismatch in argument 1: int vs Tensor"
      );

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_mismatched_with_different_num_returns_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 0 vs 1"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 2 vs 1"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 1 vs 0"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 2 vs 0"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 0 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 1 vs 2"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
        }, "The number of returns is different. 3 vs 2"
      );

    */
}

#[test] fn operator_registration_test_functor_based_kernel_given_mismatched_with_different_return_types_when_registering_then_fails() {
    todo!();
    /*
    
      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));
        }, "Type mismatch in return 1: Tensor vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<i64, Tensor>>(DispatchKey::CPU));
        }, "Type mismatch in return 1: float vs int"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(DispatchKey::CPU));
        }, "Type mismatch in return 1: float vs Tensor"
      );

      // assert this does not fail because it matches
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, i64>, Tensor>>(DispatchKey::CPU));

      // and now a set of mismatching schemas
      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, i64>, Tensor>>(DispatchKey::CPU));
        }, "Type mismatch in return 2: float vs int"
      );

      expectThrows<c10::Error>([] {
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, i64>, Tensor>>(DispatchKey::CPU));
        }, "Type mismatch in return 1: int vs Tensor"
      );

    */
}
