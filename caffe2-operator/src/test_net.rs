crate::ix!();

#[test] fn net_test_disabled_chaining_for_different_devices() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out"
              type: "NetTestDummy"
              device_option {
                device_type: 1
              }
            }
            op {
              input: "out"
              output: "out2"
              type: "NetTestDummy"
              device_option {
                device_type: 1
              }
            }
            op {
              input: "out2"
              output: "out3"
              type: "NetTestDummy"
              device_option {
                device_type: 1
                device_id: 1
              }
            }
    )DOC";
      if (HasCudaGPU() && NumCudaDevices() >= 2) {
        checkChainingAndRun(spec, {{0, {0, 1, 2}}, {3, {3}}});
      }
  */
}

#[test] fn net_test_construction_no_declared_input_output() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      unique_ptr<NetBase> net(
          CreateNetTestHelper(&ws, vector<string>(), vector<string>()));
      EXPECT_TRUE(net.get() != nullptr);
  */
}

#[test] fn net_test_construction_declared_input() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      unique_ptr<NetBase> net(
          CreateNetTestHelper(&ws, vector<string>{"in"}, vector<string>()));
      EXPECT_TRUE(net.get() != nullptr);
  */
}

#[test] fn net_test_construction_declared_output() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      unique_ptr<NetBase> net(
          CreateNetTestHelper(&ws, vector<string>(), vector<string>{"out"}));
      EXPECT_TRUE(net.get() != nullptr);
  */
}

#[test] fn net_test_declared_input_insufficient() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      ASSERT_THROW(
          CreateNetTestHelper(&ws, vector<string>{"unuseful_in"}, vector<string>()),
          EnforceNotMet);
  */
}

#[test] fn net_death_test_declared_output_not_met() {
    todo!();
    /*
      Workspace ws;
      ws.CreateBlob("in");
      ASSERT_THROW(
          CreateNetTestHelper(
              &ws, vector<string>(), vector<string>{"unproduced_out"}),
          EnforceNotMet);
  */
}

#[inline] pub fn test_execution(
    net: &mut Box<NetBase>,
    num_ops: i32)
{
    todo!();
    /*
        // Run 100 times
      for (int i = 0; i < 100; i++) {
        counter.exchange(0);
        net.get()->Run();
        ASSERT_EQ(num_ops, counter.load());
      }
    */
}

#[test] fn net_test_disabled_chaining_for_linear_model() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out"
              type: "NetTestDummy"
            }
    )DOC";
      checkChainingAndRun(spec, {{0, {0, 1}}});
  */
}

#[test] fn net_test_disabled_chaining_for_fork() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out1"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out2"
              type: "NetTestDummy"
            }
    )DOC";
      checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2}}});
  */
}

/**
  | TEST(NetTest, ChainingForJoinWithAncestor) {
  |   const auto spec = R"DOC(
  |         name: "example"
  |         type: "dag"
  |         external_input: "in"
  |         op {
  |           input: "in"
  |           output: "hidden"
  |           type: "NetTestDummy"
  |         }
  |         op {
  |           input: "hidden"
  |           output: "out1"
  |           type: "NetTestDummy"
  |         }
  |         op {
  |           input: "hidden"
  |           output: "out2"
  |           type: "NetTestDummy"
  |         }
  |         op {
  |           input: "hidden"
  |           input: "out2"
  |           type: "NetTestDummy"
  |         }
  | )DOC";
  |   checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
  | }
  */
#[test] fn net_test_disabled_chaining_for_fork_join() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden1"
              type: "NetTestDummy"
            }
            op {
              input: "in"
              output: "hidden2"
              type: "NetTestDummy"
            }
            op {
              input: "hidden1"
              input: "hidden2"
              output: "out"
              type: "NetTestDummy"
            }
            op {
              input: "out"
              output: "out2"
              type: "NetTestDummy"
            }
    )DOC";
      checkChainingAndRun(spec, {{0, {0}}, {1, {1}}, {2, {2, 3}}});
  */
}


#[test] fn net_test_disabled_chaining_forward_backward() {
    todo!();
    /*
      const auto spec = R"DOC(
      name: "gpu_0"
      type: "dag"
      op {
        input: "in"
        input: "fc_0_w"
        input: "fc_0_b"
        output: "fc_0"
        name: "0"
        type: "NetTestDummy"
      }
      op {
        input: "fc_0"
        output: "fc_0"
        name: "1"
        type: "NetTestDummy"
      }
      op {
        input: "fc_0"
        input: "fc_1_w"
        input: "fc_1_b"
        output: "fc_1"
        name: "2"
        type: "NetTestDummy"
      }
      op {
        input: "fc_1"
        output: "fc_1"
        name: "3"
        type: "NetTestDummy"
      }
      op {
        input: "fc_1"
        input: "fc_2_w"
        input: "fc_2_b"
        output: "fc_2"
        name: "4"
        type: "NetTestDummy"
      }
      op {
        input: "fc_2"
        output: "fc_2"
        name: "5"
        type: "NetTestDummy"
      }
      op {
        input: "fc_2"
        input: "fc_3_w"
        input: "fc_3_b"
        output: "fc_3"
        name: "6"
        type: "NetTestDummy"
      }
      op {
        input: "fc_3"
        output: "fc_3"
        name: "7"
        type: "NetTestDummy"
      }
      op {
        input: "fc_3"
        input: "fc_4_w"
        input: "fc_4_b"
        output: "fc_4"
        name: "8"
        type: "NetTestDummy"
      }
      op {
        input: "fc_4"
        output: "fc_4"
        name: "9"
        type: "NetTestDummy"
      }
      op {
        input: "fc_4"
        input: "in2"
        output: "LabelCrossEntropy"
        name: "10"
        type: "NetTestDummy"
      }
      op {
        input: "LabelCrossEntropy"
        output: "AveragedLoss"
        name: "11"
        type: "NetTestDummy"
      }
      op {
        input: "AveragedLoss"
        output: "AveragedLoss_autogen_grad"
        name: "12"
        type: "NetTestDummy"
      }
      op {
        input: "LabelCrossEntropy"
        input: "AveragedLoss_autogen_grad"
        output: "LabelCrossEntropy_grad"
        name: "13"
        type: "NetTestDummy"
      }
      op {
        input: "fc_4"
        input: "label"
        input: "LabelCrossEntropy_grad"
        output: "fc_4_grad"
        name: "14"
        type: "NetTestDummy2"
      }
      op {
        input: "fc_4"
        input: "fc_4_grad"
        output: "fc_4_grad"
        name: "15"
        type: "NetTestDummy2"
      }
      op {
        input: "fc_3"
        input: "fc_4_w"
        input: "fc_4_grad"
        output: "fc_4_w_grad"
        output: "fc_4_b_grad"
        output: "fc_3_grad"
        name: "16"
        type: "NetTestDummy"
      }
      op {
        input: "fc_3"
        input: "fc_3_grad"
        output: "fc_3_grad"
        name: "17"
        type: "NetTestDummy2"
      }
      op {
        input: "fc_2"
        input: "fc_3_w"
        input: "fc_3_grad"
        output: "fc_3_w_grad"
        output: "fc_3_b_grad"
        output: "fc_2_grad"
        name: "18"
        type: "NetTestDummy"
      }
      op {
        input: "fc_2"
        input: "fc_2_grad"
        output: "fc_2_grad"
        name: "19"
        type: "NetTestDummy2"
      }
      op {
        input: "fc_1"
        input: "fc_2_w"
        input: "fc_2_grad"
        output: "fc_2_w_grad"
        output: "fc_2_b_grad"
        output: "fc_1_grad"
        name: "20"
        type: "NetTestDummy"
      }
      op {
        input: "fc_1"
        input: "fc_1_grad"
        output: "fc_1_grad"
        name: "21"
        type: "NetTestDummy2"
      }
      op {
        input: "fc_0"
        input: "fc_1_w"
        input: "fc_1_grad"
        output: "fc_1_w_grad"
        output: "fc_1_b_grad"
        output: "fc_0_grad"
        name: "22"
        type: "NetTestDummy"
      }
      op {
        input: "fc_0"
        input: "fc_0_grad"
        output: "fc_0_grad"
        name: "23"
        type: "NetTestDummy2"
      }
      op {
        input: "in"
        input: "fc_0_w"
        input: "fc_0_grad"
        output: "fc_0_w_grad"
        output: "fc_0_b_grad"
        output: "data_grad"
        name: "24"
        type: "NetTestDummy"
      }
      external_input: "in"
      external_input: "in2"
      external_input: "LR"
      external_input: "fc_0_w"
      external_input: "fc_0_b"
      external_input: "fc_1_w"
      external_input: "fc_1_b"
      external_input: "fc_2_w"
      external_input: "fc_2_b"
      external_input: "fc_3_w"
      external_input: "fc_3_b"
      external_input: "fc_4_w"
      external_input: "fc_4_b"
      external_input: "label"
      )DOC";
      checkNumChainsAndRun(spec, 1);
  */
}


#[test] fn net_test_disabled_chaining_for_hogwild_model() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden1"
              type: "NetTestDummy"
            }
            op {
              input: "hidden1"
              output: "mid1"
              type: "NetTestDummy"
            }
            op {
              input: "mid1"
              output: "out1"
              type: "NetTestDummy"
            }
            op {
              input: "in"
              output: "hidden2"
              type: "NetTestDummy"
            }
            op {
              input: "hidden2"
              output: "mid2"
              type: "NetTestDummy"
            }
            op {
              input: "mid2"
              output: "out2"
              type: "NetTestDummy"
            }
    )DOC";
      checkNumChainsAndRun(spec, 2);
  */
}


#[test] fn net_test_disabled_failing_operator() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "NetTestDummy"
            }
            op {
              input: "hidden"
              output: "out"
              type: "NetTestDummy"
              arg {
                name: "fail"
                i: 1
              }
            }
    )DOC";

      Workspace ws;
      ws.CreateBlob("in");

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        net_def.set_num_workers(4);
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        for (int i = 0; i < 10; i++) {
          counter.exchange(0);
          bool run_result = false;
          try {
            run_result = net->Run();
          } catch (const std::exception&) {
            // async_scheduling would throw
          }
          ASSERT_FALSE(run_result);

          ASSERT_EQ(1, counter.load());
        }
      }
  */
}

#[test] fn net_test_operator_with_executor_helper() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "async_scheduling"
            op {
              type: "ExecutorHelperDummy"
            }
    )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      Workspace ws;
      net_def.set_num_workers(kTestPoolSize);
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      ASSERT_TRUE(net->Run());
  */
}


#[test] fn net_test_disabled_operator_with_disabled_event() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "async_scheduling"
            external_input: "in"
            op {
              input: "in"
              output: "out"
              type: "NetTestDummy"
              arg {
                name: "fail"
                i: 1
              }
            }
    )DOC";

      Workspace ws;
      ws.CreateBlob("in");

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        net->GetOperators()[0]->DisableEvent();
        // async_scheduling propagates exception
        bool caught_exception = false;
        try {
          net->Run();
        } catch (const std::exception& e) {
          caught_exception = true;
        }
        ASSERT_TRUE(caught_exception);
      }
  */
}


#[test] fn net_test_executor_override() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "dag"
      )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        Workspace ws;
        auto old = FLAGS_caffe2_override_executor;
        auto g = MakeGuard([&]() { FLAGS_caffe2_override_executor = old; });
        FLAGS_caffe2_override_executor = "dag,async_scheduling";

        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        auto async_net =
            caffe2::dynamic_cast_if_rtti<AsyncSchedulingNet*>(net.get());
        ASSERT_TRUE(async_net != nullptr);
      }
  */
}


#[test] fn net_test_async_empty_net() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "async_scheduling"
      )DOC";

      Workspace ws;
      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        bool caught_exception = false;
        try {
          ASSERT_TRUE(net->Run());
        } catch (const std::exception& e) {
          caught_exception = true;
        }
        ASSERT_FALSE(caught_exception);
      }
  */
}


#[test] fn net_test_disabled_run_async_failure() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "async_scheduling"
            op {
              input: "in"
              output: "out"
              type: "NetTestDummy"
              arg {
                name: "fail"
                i: 1
              }
            }
      )DOC";

      Workspace ws;
      ws.CreateBlob("in");

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));

        bool caught_exception = false;
        try {
          ASSERT_FALSE(net->Run());
        } catch (const std::exception& e) {
          caught_exception = true;
        }
        ASSERT_TRUE(caught_exception);
      }
  */
}


#[test] fn net_test_no_type_net() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "no_type_net"
      )DOC";

      Workspace ws;
      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      {
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        ASSERT_TRUE(net);
      }
  */
}

#[test] fn net_test_pending_ops_and_net_failure() {
    todo!();
    /*
      const auto spec = R"DOC(
            name: "example"
            type: "async_scheduling"
            op {
              type: "NotFinishingOp"
            }
            op {
              type: "NetTestDummy"
              arg {
                name: "fail"
                i: 1
              }
            }
    )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      Workspace ws;
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));

      try {
        // net is not stuck and returns false
        ASSERT_FALSE(net->Run());
      } catch (const caffe2::AsyncNetCancelled&) {
        // Cancellation exception is fine since if the ops run concurrently the
        // NotFinishingOp may be cancelled with an exception.
      }
  */
}

#[test] fn net_test_async_error_op_test() {

    todo!();
    /*
      Workspace ws;

      // Throw in sync part
      auto net = AsyncErrorNet(&ws, "net1", /*throw_*/ true, /*fail_in_sync*/ true);
    #ifdef CAFFE2_USE_EXCEPTION_PTR
      ASSERT_THROW(net->Run(), std::logic_error);
    #endif

      // Return false in sync part
      net = AsyncErrorNet(&ws, "net2", /*throw_*/ false, /*fail_in_sync*/ true);
      ASSERT_FALSE(net->Run());

      // SetFinishedWithException in async part
      net = AsyncErrorNet(&ws, "net3", /*throw_*/ true, /*fail_in_sync*/ false);
    #ifdef CAFFE2_USE_EXCEPTION_PTR
      ASSERT_THROW(net->Run(), std::logic_error);
    #endif

      // SetFinished(err) in async part
      net = AsyncErrorNet(&ws, "net4", /*throw_*/ false, /*fail_in_sync*/ false);
      ASSERT_FALSE(net->Run());
  */
}

#[test] fn net_test_async_error_timings_test() {
    todo!();
    /*
      Workspace ws;
      std::string spec = R"DOC(
            name: "net"
            type: "async_scheduling"
            op {
              type: "AsyncErrorOp"
              arg {
                name: "throw"
                i: 1
              }
              arg {
                name: "fail_in_sync"
                i: 0
              }
              arg {
                name: "sleep_time"
                i: 2
              }
              arg {
                name: "error_msg"
                s: "Error1"
              }
            }
            op {
              type: "AsyncErrorOp"
              arg {
                name: "throw"
                i: 1
              }
              arg {
                name: "fail_in_sync"
                i: 0
              }
              arg {
                name: "sleep_time"
                i: 1
              }
              arg {
                name: "error_msg"
                s: "Error2"
              }
            }
      )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      auto net = CreateNet(net_def, &ws);

      try {
        net->Run();
      } catch (const std::logic_error& e) {
        ASSERT_TRUE(std::string(e.what()) == "Error2");
      } catch (...) {
        FAIL() << "Expected std::logic_error thrown";
      }
  */
}

#[test] fn net_test_chain_error_test() {
    todo!();
    /*
      Workspace ws;

      auto net = ChainErrorNet(&ws, "net1", /*throw_*/ true);
    #ifdef CAFFE2_USE_EXCEPTION_PTR
      ASSERT_THROW(net->Run(), std::logic_error);
    #endif

      net = ChainErrorNet(&ws, "net2", /*throw_*/ false);
      ASSERT_FALSE(net->Run());
  */
}

#[inline] pub fn test_prof_dagnet_error_case(test_error: bool)  {
    
    todo!();
    /*
        std::string spec_template = R"DOC(
            name: "prof_dag_error_test_net"
            type: "prof_dag"
            external_input: "in"
            op {
              input: "in"
              output: "hidden"
              type: "SyncErrorOp"
              arg {
                name: "fail"
                i: <FAIL>
              }
              arg {
                name: "throw"
                i: 0
              }
            }
            op {
              input: "hidden"
              output: "out"
              type: "SyncErrorOp"
              arg {
                name: "fail"
                i: 0
              }
            }
      )DOC";

      Workspace ws;
      ws.CreateBlob("in");

      NetDef net_def;
      std::string net_spec = spec_template;
      ReplaceAll(net_spec, "<FAIL>", test_error ? "1" : "0");
      CAFFE_ENFORCE(TextFormat::ParseFromString(net_spec, &net_def));
      auto net = CreateNet(net_def, &ws);

      // with failing op - net runs return false, without - true
      for (auto num_runs = 0; num_runs < 10; ++num_runs) {
        auto ret = net->Run();
        ASSERT_TRUE(test_error ? !ret : ret);
      }

      // with failing op - prof_dag handles invalid runs and returns empty stats,
      // without - returns stats for each op
      auto* prof_dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
      CHECK_NOTNULL(prof_dag);
      auto stats_proto = prof_dag->GetPerOperatorCost();
      ASSERT_EQ(
          stats_proto.stats_size(), test_error ? 0 : net->GetOperators().size());
    */
}

#[test] fn net_test_prof_dag_net_error_test() {
    todo!();
    /*
      testProfDAGNetErrorCase(/*test_error=*/false);
      testProfDAGNetErrorCase(/*test_error=*/true);
  */
}
