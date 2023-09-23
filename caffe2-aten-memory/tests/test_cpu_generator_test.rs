crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/cpu_generator_test.cpp]

#[test] fn cpu_generator_impl_test_dynamic_cast() {
    todo!();
    /*
    
      // Test Description: Check dynamic cast for CPU
      auto foo = createCPUGenerator();
      auto result = check_generator<CPUGeneratorImpl>(foo);
      ASSERT_EQ(typeid(CPUGeneratorImpl*).hash_code(), typeid(result).hash_code());

    */
}

#[test] fn cpu_generator_impl_test_default() {
    todo!();
    /*
    
      // Test Description:
      // Check if default generator is created only once
      // address of generator should be same in all calls
      auto foo = getDefaultCPUGenerator();
      auto bar = getDefaultCPUGenerator();
      ASSERT_EQ(foo, bar);

    */
}

#[test] fn cpu_generator_impl_test_cloning() {
    todo!();
    /*
    
      // Test Description:
      // Check cloning of new generators.
      // Note that we don't allow cloning of other
      // generator states into default generators.
      auto gen1 = createCPUGenerator();
      auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
      cpu_gen1->random(); // advance gen1 state
      cpu_gen1->random();
      auto gen2 = createCPUGenerator();
      gen2 = gen1.clone();
      auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);
      ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());

    */
}

pub fn thread_func_get_engine_op(generator: *mut CPUGeneratorImpl)  {
    
    todo!();
        /*
            lock_guard<mutex> lock(generator->mutex_);
      generator->random();
        */
}

#[test] fn cpu_generator_impl_test_multithreading_get_engine_operator() {
    todo!();
    /*
    
      // Test Description:
      // Check CPUGeneratorImpl is reentrant and the engine state
      // is not corrupted when multiple threads request for
      // random samples.
      // See Note [Acquire lock when using random generators]
      auto gen1 = createCPUGenerator();
      auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
      auto gen2 = createCPUGenerator();
      {
        lock_guard<mutex> lock(gen1.mutex());
        gen2 = gen1.clone(); // capture the current state of default generator
      }
      thread t0{thread_func_get_engine_op, cpu_gen1};
      thread t1{thread_func_get_engine_op, cpu_gen1};
      thread t2{thread_func_get_engine_op, cpu_gen1};
      t0.join();
      t1.join();
      t2.join();
      lock_guard<mutex> lock(gen2.mutex());
      auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);
      cpu_gen2->random();
      cpu_gen2->random();
      cpu_gen2->random();
      ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());

    */
}

#[test] fn cpu_generator_impl_test_get_set_current_seed() {
    todo!();
    /*
    
      // Test Description:
      // Test current seed getter and setter
      // See Note [Acquire lock when using random generators]
      auto foo = getDefaultCPUGenerator();
      lock_guard<mutex> lock(foo.mutex());
      foo.set_current_seed(123);
      auto current_seed = foo.current_seed();
      ASSERT_EQ(current_seed, 123);

    */
}

pub fn thread_func_get_set_current_seed(generator: Generator)  {
    
    todo!();
        /*
            lock_guard<mutex> lock(generator.mutex());
      auto current_seed = generator.current_seed();
      current_seed++;
      generator.set_current_seed(current_seed);
        */
}

#[test] fn cpu_generator_impl_test_multithreading_get_set_current_seed() {
    todo!();
    /*
    
      // Test Description:
      // Test current seed getter and setter are thread safe
      // See Note [Acquire lock when using random generators]
      auto gen1 = getDefaultCPUGenerator();
      auto initial_seed = gen1.current_seed();
      thread t0{thread_func_get_set_current_seed, gen1};
      thread t1{thread_func_get_set_current_seed, gen1};
      thread t2{thread_func_get_set_current_seed, gen1};
      t0.join();
      t1.join();
      t2.join();
      ASSERT_EQ(gen1.current_seed(), initial_seed+3);

    */
}

#[test] fn cpu_generator_impl_test_rng_forking() {
    todo!();
    /*
    
      // Test Description:
      // Test that state of a generator can be frozen and
      // restored
      // See Note [Acquire lock when using random generators]
      auto default_gen = getDefaultCPUGenerator();
      auto current_gen = createCPUGenerator();
      {
        lock_guard<mutex> lock(default_gen.mutex());
        current_gen = default_gen.clone(); // capture the current state of default generator
      }
      auto target_value = randn({1000});
      // Dramatically alter the internal state of the main generator
      auto x = randn({100000});
      auto forked_value = randn({1000}, current_gen);
      ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());

    */
}

/* ------------ * Philox CPU Engine Tests  ------------ */

#[test] fn cpu_generator_impl_test_philox_engine_reproducibility() {
    todo!();
    /*
    
      // Test Description:
      //   Tests if same inputs give same results.
      //   launch on same thread index and create two engines.
      //   Given same seed, idx and offset, assert that the engines
      //   should be aligned and have the same sequence.
      Philox4_32_10 engine1(0, 0, 4);
      Philox4_32_10 engine2(0, 0, 4);
      ASSERT_EQ(engine1(), engine2());

    */
}

#[test] fn cpu_generator_impl_test_philox_engine_offset1() {
    todo!();
    /*
    
      // Test Description:
      //   Tests offsetting in same thread index.
      //   make one engine skip the first 8 values and
      //   make another engine increment to until the
      //   first 8 values. Assert that the first call
      //   of engine2 and the 9th call of engine1 are equal.
      Philox4_32_10 engine1(123, 1, 0);
      // Note: offset is a multiple of 4.
      // So if you want to skip 8 values, offset would
      // be 2, since 2*4=8.
      Philox4_32_10 engine2(123, 1, 2);
      for(int i = 0; i < 8; i++){
        // Note: instead of using the engine() call 8 times
        // we could have achieved the same functionality by
        // calling the incr() function twice.
        engine1();
      }
      ASSERT_EQ(engine1(), engine2());

    */
}

#[test] fn cpu_generator_impl_test_philox_engine_offset2() {
    todo!();
    /*
    
      // Test Description:
      //   Tests edge case at the end of the 2^190th value of the generator.
      //   launch on same thread index and create two engines.
      //   make engine1 skip to the 2^64th 128 bit while being at thread 0
      //   make engine2 skip to the 2^64th 128 bit while being at 2^64th thread
      //   Assert that engine2 should be increment_val+1 steps behind engine1.
      unsigned long long increment_val = u64::max;
      Philox4_32_10 engine1(123, 0, increment_val);
      Philox4_32_10 engine2(123, increment_val, increment_val);

      engine2.incr_n(increment_val);
      engine2.incr();
      ASSERT_EQ(engine1(), engine2());

    */
}

#[test] fn cpu_generator_impl_test_philox_engine_offset3() {
    todo!();
    /*
    
      // Test Description:
      //   Tests edge case in between thread indices.
      //   launch on same thread index and create two engines.
      //   make engine1 skip to the 2^64th 128 bit while being at thread 0
      //   start engine2 at thread 1, with offset 0
      //   Assert that engine1 is 1 step behind engine2.
      unsigned long long increment_val = u64::max;
      Philox4_32_10 engine1(123, 0, increment_val);
      Philox4_32_10 engine2(123, 1, 0);
      engine1.incr();
      ASSERT_EQ(engine1(), engine2());

    */
}

#[test] fn cpu_generator_impl_test_philox_engine_index() {
    todo!();
    /*
    
      // Test Description:
      //   Tests if thread indexing is working properly.
      //   create two engines with different thread index but same offset.
      //   Assert that the engines have different sequences.
      Philox4_32_10 engine1(123456, 0, 4);
      Philox4_32_10 engine2(123456, 1, 4);
      ASSERT_NE(engine1(), engine2());

    */
}

/* ----------- * MT19937 CPU Engine Tests  ----------- */

#[test] fn cpu_generator_impl_test_mt19937engine_reproducibility() {
    todo!();
    /*
    
      // Test Description:
      //   Tests if same inputs give same results when compared
      //   to std.

      // test with zero seed
      mt19937 engine1(0);
      mt19937 engine2(0);
      for(int i = 0; i < 10000; i++) {
        ASSERT_EQ(engine1(), engine2());
      }

      // test with large seed
      engine1 = mt19937(2147483647);
      engine2 = mt19937(2147483647);
      for(int i = 0; i < 10000; i++) {
        ASSERT_EQ(engine1(), engine2());
      }

      // test with random seed
      random_device rd;
      auto seed = rd();
      engine1 = mt19937(seed);
      engine2 = mt19937(seed);
      for(int i = 0; i < 10000; i++) {
        ASSERT_EQ(engine1(), engine2());
      }

    */
}
