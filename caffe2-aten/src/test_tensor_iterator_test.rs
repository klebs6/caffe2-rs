crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/tensor_iterator_test.cpp]

/**
  | An operation with a CUDA tensor and CPU scalar
  | should keep the scalar on the CPU (and lift it
  | to a parameter).
  |
  */
#[test] fn tensor_iterator_test_cpu_scalar() {
    todo!();
    /*
    
      if (!hasCUDA()) return;
      Tensor out;
      auto x = randn({5, 5}, kCUDA);
      auto y = ones(1, kCPU).squeeze();
      auto iter = TensorIterator::binary_op(out, x, y);
      EXPECT_TRUE(iter.device(0).is_cuda()) << "result should be CUDA";
      EXPECT_TRUE(iter.device(1).is_cuda()) << "x should be CUDA";
      EXPECT_TRUE(iter.device(2).is_cpu()) << "y should be CPU";

    */
}

/**
  | Verifies multiple zero-dim CPU inputs
  | are not coerced to CUDA
  |
  */
#[test] fn tensor_iterator_test_cpu_scalar_inputs() {
    todo!();
    /*
    
      if (!hasCUDA()) return;
      Tensor out = empty({5, 5}, kCUDA);
      auto x = ones(1, kCPU).squeeze();
      auto y = ones(1, kCPU).squeeze();
      ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));

    */
}

/**
  | Mixing CPU and CUDA tensors should raise
  | an exception (if the CPU tensor isn't
  | zero-dim)
  |
  */
#[test] fn tensor_iterator_test_mixed_devices() {
    todo!();
    /*
    
      if (!hasCUDA()) return;
      Tensor out;
      auto x = randn({5, 5}, kCUDA);
      auto y = ones({5}, kCPU);
      ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));

    */
}

pub fn random_tensor_for_type(scalar_type: ScalarType) -> Tensor {
    
    todo!();
        /*
            if (isFloatingType(scalar_type)) {
        return randn({5, 5}, device(kCPU).dtype(scalar_type));
      } else if (scalar_type == kBool) {
        return randint(0, 2, {5, 5}, device(kCPU).dtype(scalar_type));
      } else {
        return randint(1, 10, {5, 5}, device(kCPU).dtype(scalar_type));
      }
        */
}

#[macro_export] macro_rules! unary_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopUnary_##name) {                              
          Tensor out;                                                                   
          auto in = random_tensor_for_type(k##name);                                    
          auto expected = in.add(1);                                                    
          auto iter = TensorIterator::unary_op(out, in);                                
          native::cpu_serial_kernel(iter, [=](ctype a) -> ctype { return a + 1; }); 
          ASSERT_ANY_THROW(out.equal(expected));                                        
        }
        */
    }
}

#[macro_export] macro_rules! no_output_unary_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopUnaryNoOutput_##name) {                     
          auto in = random_tensor_for_type(k##name);                                   
          auto iter = TensorIteratorConfig()                                       
              .add_owned_input(in)                                                           
              .build();                                                                
          i64 acc = 0;                                                             
          native::cpu_serial_kernel(iter, [&](ctype a) -> void { acc++; }); 
          EXPECT_TRUE(acc == in.numel());                                              
        }
        */
    }
}

#[macro_export] macro_rules! binary_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopBinary_##name) {                                      
          Tensor out;                                                                            
          auto in1 = random_tensor_for_type(k##name);                                            
          auto in2 = random_tensor_for_type(k##name);                                            
          auto expected = in1.add(in2);                                                          
          auto iter = TensorIterator::binary_op(out, in1, in2);                                  
          native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> ctype { return a + b; }); 
          ASSERT_ANY_THROW(out.equal(expected));                                                 
        }
        */
    }
}

#[macro_export] macro_rules! no_output_binary_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopBinaryNoOutput_##name) {                      
          auto in1 = random_tensor_for_type(k##name);                                    
          auto in2 = random_tensor_for_type(k##name);                                    
          auto iter = TensorIteratorConfig()                                         
              .add_owned_input(in1)                                                            
              .add_owned_input(in2)                                                            
              .build();                                                                  
          i64 acc = 0;                                                               
          native::cpu_serial_kernel(iter, [&](ctype a, ctype b) -> void { acc++; }); 
          EXPECT_TRUE(acc == in1.numel());                                               
        }
        */
    }
}

#[macro_export] macro_rules! pointwise_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopPointwise_##name) {                                                
          Tensor out;                                                                                         
          auto in1 = random_tensor_for_type(k##name);                                                         
          auto in2 = random_tensor_for_type(k##name);                                                         
          auto in3 = random_tensor_for_type(k##name);                                                         
          auto expected = in1.add(in2).add(in3);                                                              
          auto iter = TensorIteratorConfig()                                                              
              .add_output(out)                                                                                
              .add_owned_input(in1)                                                                                 
              .add_owned_input(in2)                                                                                 
              .add_owned_input(in3)                                                                                 
              .build();                                                                                       
          native::cpu_serial_kernel(iter, [=](ctype a, ctype b, ctype c) -> ctype { return a + b + c; }); 
          ASSERT_ANY_THROW(out.equal(expected));                                                              
        }
        */
    }
}

#[macro_export] macro_rules! no_output_pointwise_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, SerialLoopPoinwiseNoOutput_##name) {                             
          auto in1 = random_tensor_for_type(k##name);                                             
          auto in2 = random_tensor_for_type(k##name);                                             
          auto in3 = random_tensor_for_type(k##name);                                             
          auto iter = TensorIteratorConfig()                                                  
              .add_owned_input(in1)                                                                     
              .add_owned_input(in2)                                                                     
              .add_owned_input(in3)                                                                     
              .build();                                                                           
          i64 acc = 0;                                                                        
          native::cpu_serial_kernel(iter, [&](ctype a, ctype b, ctype c) -> void { acc++; }); 
          EXPECT_TRUE(acc == in1.numel());                                                        
        }
        */
    }
}

/**
  | The alternative way to calculate a < b is (b - a).clamp(0).toBool()
  |
  | To prevent an overflow in subtraction (b - a)
  | for unsigned types(unit, bool) we will convert
  | in to int first
  |
  */
#[macro_export] macro_rules! comparison_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, ComparisonLoopBinary_##name) {                                    
          auto in1 = random_tensor_for_type(k##name);                                              
          auto in2 = random_tensor_for_type(k##name);                                              
          Tensor out = empty({0}, in1.options().dtype(kBool));                                 
          Tensor diff;                                                                             
          if (k##name == kByte || k##name == kBool) {                                              
            diff = in2.to(kInt).sub(in1.to(kInt));                                                 
          } else {                                                                                 
            diff = in2.sub(in1);                                                                   
          }                                                                                        
          auto expected = diff.clamp_min(0).to(kBool);                                             
          auto iter = TensorIterator::comparison_op(out, in1, in2);                                
          native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> bool { return a < b; });    
          EXPECT_TRUE(out.equal(expected));                                                        
        }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types!{UNARY_TEST_ITER_FOR_TYPE}
    at_forall_scalar_types!{BINARY_TEST_ITER_FOR_TYPE}
    at_forall_scalar_types!{POINTWISE_TEST_ITER_FOR_TYPE}
    at_forall_scalar_types!{NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE}
    at_forall_scalar_types!{NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE}
    at_forall_scalar_types!{NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE}
    */
}

lazy_static!{
    /*
    at_forall_scalar_types_and!{Bool, COMPARISON_TEST_ITER_FOR_TYPE}
    */
}

#[test] fn tensor_iterator_test_serial_loop_single_thread() {
    todo!();
    /*
    
      thread::id thread_id = this_thread::get_id();
      Tensor out;
      auto x = zeros({50000}, TensorOptions(kCPU).dtype(kInt));
      auto iter = TensorIterator::unary_op(out, x);
      native::cpu_serial_kernel(iter, [=](int a) -> int {
        thread::id lambda_thread_id = this_thread::get_id();
        EXPECT_TRUE(lambda_thread_id == thread_id);
        return a + 1;
      });

    */
}

#[test] fn tensor_iterator_test_input_dtype() {
    todo!();
    /*
    
      auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_owned_output(ones({1, 1}, dtype(kBool)))
          .add_owned_input(ones({1, 1}, dtype(kFloat)))
          .add_owned_input(ones({1, 1}, dtype(kDouble)))
          .build();
      EXPECT_TRUE(iter.input_dtype() == kFloat);
      EXPECT_TRUE(iter.input_dtype(0) == kFloat);
      EXPECT_TRUE(iter.input_dtype(1) == kDouble);

    */
}

#[test] fn tensor_iterator_test_compute_common_dtype_input_only() {
    todo!();
    /*
    
      auto iter = TensorIteratorConfig()
          .add_owned_output(ones({1, 1}, dtype(kBool)))
          .add_owned_input(ones({1, 1}, dtype(kFloat)))
          .add_owned_input(ones({1, 1}, dtype(kDouble)))
          .promote_inputs_to_common_dtype(true)
          .build();
      EXPECT_TRUE(iter.dtype(0) == kBool);
      EXPECT_TRUE(iter.dtype(1) == kDouble);
      EXPECT_TRUE(iter.dtype(2) == kDouble);
      EXPECT_TRUE(iter.common_dtype() == kDouble);

    */
}

#[test] fn tensor_iterator_test_do_not_compute_common_dtype_input_only() {
    todo!();
    /*
    
      auto iter = TensorIteratorConfig()
          .check_all_same_dtype(false)
          .add_owned_output(ones({1, 1}, dtype(kLong)))
          .add_owned_input(ones({1, 1}, dtype(kFloat)))
          .add_owned_input(ones({1, 1}, dtype(kDouble)))
          .build();
      EXPECT_TRUE(iter.dtype(0) == kLong);
      EXPECT_TRUE(iter.dtype(1) == kFloat);
      EXPECT_TRUE(iter.dtype(2) == kDouble);

    */
}

#[test] fn tensor_iterator_test_fail_non_promoting_binary_op() {
    todo!();
    /*
    
      Tensor out;
      TensorIteratorConfig config;
      config.add_output(out);
      config.add_owned_input(ones({1,1}, dtype(kDouble)));
      config.add_owned_input(ones({1,1}, dtype(kInt)));
      ASSERT_ANY_THROW(config.build());

    */
}

#[macro_export] macro_rules! multiple_outputs_test_iter_for_type {
    ($ctype:ident, $name:ident) => {
        /*
        
        TEST(TensorIteratorTest, CpuKernelMultipleOutputs_##name) {                                         
          auto in1 = random_tensor_for_type(k##name);                                                       
          auto in2 = random_tensor_for_type(k##name);                                                       
          Tensor out1 = empty({0}, in1.options());                                                      
          Tensor out2 = empty({0}, in1.options());                                                      
          auto expected1 = in1.add(in2);                                                                    
          auto expected2 = in1.mul(in2);                                                                    
          auto iter = TensorIteratorConfig()                                                            
            .add_output(out1)                                                                               
            .add_output(out2)                                                                               
            .add_owned_input(in1)                                                                                 
            .add_owned_input(in2)                                                                                 
            .build();                                                                                       
          native::cpu_kernel_multiple_outputs(iter, [=](ctype a, ctype b) -> tuple<ctype, ctype> { 
            ctype add = a + b;                                                                              
            ctype mul = a * b;                                                                              
            return tuple<ctype, ctype>(add, mul);                                                      
          });                                                                                               
          EXPECT_TRUE(out1.equal(expected1));                                                               
          EXPECT_TRUE(out2.equal(expected2));                                                               
        }
        */
    }
}

lazy_static!{
    /*
    at_forall_scalar_types!{MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE}
    */
}

