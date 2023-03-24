crate::ix!();

/**
  | When measuring time, we relax the measured
  | time by +- 40ms.
  |
  */
#[cfg(not(win32))]
pub const kTimeThreshold: i32 = 40;

/// Even more so on Windows
#[cfg(win32)]
pub const kTimeThreshold: i32 = 50;

/**
  | Run a network and get its duration in
  | milliseconds.
  |
  */
#[inline] pub fn run_net_and_get_duration(net_def_str: &String, ty: &String) -> i32 {
    
    todo!();
    /*
        NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(net_def_str, &net_def));
      net_def.set_type(type);
      Workspace ws;
      unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      CAFFE_ENFORCE(net.get() != nullptr);
      // Run once to kick in potential initialization (can be slower)
      CAFFE_ENFORCE(net->Run());
      // Now run and time it
      auto start_time = std::chrono::system_clock::now();
      CAFFE_ENFORCE(net->Run());
      // Inspect the time - it should be around 200 milliseconds, since sleep3 can
      // run in parallel with sleep1 and sleep2.
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - start_time);
      int milliseconds = duration.count();
      return milliseconds;
    */
}

#[test] fn dag_net_test_dag_net_timing() {
    todo!();
    /*
      int ms = RunNetAndGetDuration(string(kSleepNetDefString), "dag");
      EXPECT_NEAR(ms, 200, kTimeThreshold);
  */
}

/**
  | For sanity check, we also test the sequential
  | time - it should take 0.35 seconds instead
  | since everything has to be sequential.
  |
  */
#[test] fn simple_net_test_simple_net_timing() {
    todo!();
    /*
      int ms = RunNetAndGetDuration(string(kSleepNetDefString), "simple");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}

#[test] fn dag_net_test_dag_net_timing_read_after_read() {
    todo!();
    /*
      int ms = RunNetAndGetDuration(string(kSleepNetDefStringReadAfterRead), "dag");
      EXPECT_NEAR(ms, 250, kTimeThreshold);
  */
}

/**
  | For sanity check, we also test the sequential
  | time - it should take 0.35 seconds instead
  | since everything has to be sequential.
  |
  */
#[test] fn simple_net_test_simple_net_timing_read_after_read() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringReadAfterRead), "simple");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}

#[test] fn dag_net_test_dag_net_timing_write_after_write() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringWriteAfterWrite), "dag");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}

#[test] fn simple_net_test_simple_net_timing_write_after_write() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringWriteAfterWrite), "simple");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}


#[test] fn dag_net_test_dag_net_timing_write_after_read() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringWriteAfterRead), "dag");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}

#[test] fn simple_net_test_simple_net_timing_write_after_read() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringWriteAfterRead), "simple");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}


#[test] fn dag_net_test_dag_net_timing_control_dependency() {
    todo!();
    /*
      int ms =
          RunNetAndGetDuration(string(kSleepNetDefStringControlDependency), "dag");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}

#[test] fn simple_net_test_simple_net_timing_control_dependency() {
    todo!();
    /*
      int ms = RunNetAndGetDuration(
          string(kSleepNetDefStringControlDependency), "simple");
      EXPECT_NEAR(ms, 350, kTimeThreshold);
  */
}
