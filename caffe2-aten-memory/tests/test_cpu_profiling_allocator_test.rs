crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cpu_profiling_allocator_test.cpp]

pub fn run_with_control_flow(
        input:         Tensor,
        conv_weight:   Tensor,
        linear_weight: Tensor,
        cond:          bool,
        pointers:      &mut Vec<*mut c_void>,
        record:        bool,
        validate:      bool) -> Tensor {

    let record: bool = record.unwrap_or(false);
    let validate: bool = validate.unwrap_or(false);

    todo!();
        /*
            if (cond) {
        input = input * 2;
      }
      void* input_ptr = input.data_ptr();
      auto conv_out = conv2d(input, conv_weight);
      void* conv_out_ptr = input.data_ptr();
      auto conv_out_flat = conv_out.view({conv_out.size(0), -1});
      auto output = linear(conv_out_flat, linear_weight);
      if (record) {
        pointers.push_back(input_ptr);
        pointers.push_back(conv_out_ptr);
      }
      if (validate) {
        TORCH_CHECK(input_ptr == pointers[0]);
        TORCH_CHECK(conv_out_ptr == pointers[1]);
      }
      return output;
        */
}

#[test] fn cpu_allocation_plan_test_with_control_flow() {
    todo!();
    /*
    
      Tensor a = rand({23, 16, 16, 16});
      Tensor conv_weight = rand({16, 16, 3, 3});
      // output shape
      // 23, 16, 14, 14
      // Flattened shape = 23, 3136
      Tensor linear_weight = rand({32, 3136});
      Tensor output, ref_output;
      vector<void*> pointers;

      auto valid_allocation_plan = [&]() {
        AllocationPlan plan;
        {
          WithProfileAllocationsGuard profile_guard(&plan);
          ref_output = run_with_control_flow(
              a, conv_weight, linear_weight, true, pointers);
        }
      };
      ASSERT_NO_THROW(valid_allocation_plan());

      auto validate_allocation_plan =
        [&](bool record_mode, bool validation_mode) -> bool {
        AllocationPlan plan;
        {
          WithProfileAllocationsGuard profile_guard(&plan);
          ref_output =
            run_with_control_flow(a, conv_weight, linear_weight, record_mode, pointers);
        }
        bool success{true};
        for (u64 i = 0; i < 10; ++i) {
          bool validation_success;
          {
            WithValidateAllocationPlanGuard
              validation_guard(&plan, &validation_success);
            output = run_with_control_flow(
                a, conv_weight, linear_weight, validation_mode, pointers);
          }
          success = success && validation_success;
        }
        return success;
      };
      ASSERT_FALSE(validate_allocation_plan(false, true));
      ASSERT_FALSE(validate_allocation_plan(true, false));
      ASSERT_TRUE(validate_allocation_plan(true, true));
      ASSERT_TRUE(validate_allocation_plan(false, false));

    */
}

#[test] fn cpu_allocation_plan_test_with_profiling_alloc() {
    todo!();
    /*
    
      Tensor a = rand({23, 16, 16, 16});
      Tensor conv_weight = rand({16, 16, 3, 3});
      // output shape
      // 23, 16, 14, 14
      // Flattened shape = 23, 3136
      Tensor linear_weight = rand({32, 3136});
      Tensor output, ref_output;
      vector<void*> pointers;

      auto valid_allocation_plan = [&]() {
        AllocationPlan plan;
        {
          WithProfileAllocationsGuard profile_guard(&plan);
          ref_output = run_with_control_flow(
              a, conv_weight, linear_weight, false, pointers);
        }
      };
      ASSERT_NO_THROW(valid_allocation_plan());

      auto validate_allocation_plan =
        [&](bool record_mode,
            bool validation_mode,
            bool validate_pointers) {
          pointers.clear();
          AllocationPlan plan;
          {
            WithProfileAllocationsGuard profile_guard(&plan);
            ref_output = run_with_control_flow(
                a,
                conv_weight,
                linear_weight,
                record_mode,
                pointers,
                false,
                false);
          }
          CPUProfilingAllocator profiling_allocator;
          {
            WithProfilingAllocatorGuard
              profiling_allocator_guard(&profiling_allocator, &plan);
            output = run_with_control_flow(
                a,
                conv_weight,
                linear_weight,
                validation_mode,
                pointers,
                validate_pointers,
                false);
          }
          for (u64 i = 0; i < 10; ++i) {
            {
              WithProfilingAllocatorGuard
                profiling_allocator_guard(&profiling_allocator, &plan);
              output = run_with_control_flow(
                  a,
                  conv_weight,
                  linear_weight,
                  validation_mode,
                  pointers,
                  false,
                  validate_pointers);
            }
          }
      };
      // When control flow conditions are same between profiling and evaluation
      // profiling allocator should not throw.
      ASSERT_NO_THROW(validate_allocation_plan(true, true, false));
      ASSERT_TRUE(ref_output.equal(output));
      ASSERT_NO_THROW(validate_allocation_plan(false, false, false));
      ASSERT_TRUE(ref_output.equal(output));
      // Furthermore profiling allocator should return the same pointers
      // back for the intermediate tensors
      ASSERT_NO_THROW(validate_allocation_plan(true, true, true));
      ASSERT_TRUE(ref_output.equal(output));
      ASSERT_NO_THROW(validate_allocation_plan(false, false, true));
      ASSERT_TRUE(ref_output.equal(output));

      // When control flow conditions are different between profiling and evaluation
      // profiling allocator should throw.
      ASSERT_THROW(validate_allocation_plan(true, false, false), Error);
      ASSERT_THROW(validate_allocation_plan(false, true, false), Error);

    */
}

pub fn main(
    argc: i32,
    argv: &[*mut u8]) -> i32 {
    
    todo!();
        /*
            // Setting the priority high to make sure no other allocator gets used instead of this.
      SetCPUAllocator(GetDefaultMobileCPUAllocator(), /*priority*/ 100);
      // Need to disable mkldnn for this test since it allocatred memory
      // via raw_allocate inteface which requires context pointer and raw
      // pointer to be the same. Tis is not true for mobile allocator.
      globalContext().setUserEnabledMkldnn(false);
      ::testing::InitGoogleTest(&argc, argv);
      manual_seed(42);
      return RUN_ALL_TESTS();
        */
}
