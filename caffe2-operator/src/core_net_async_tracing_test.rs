crate::ix!();

#[inline] pub fn test_extract_shard_id(name: &String, expected_id: i32)  {
    
    todo!();
    /*
        EXPECT_EQ(extractShardId(name), expectedId);
    */
}


#[test] fn NetAsyncTracingTest_ExtractShardId() {
    todo!();
    /*
      testExtractShardId("ABCDEFshard:1705!!A", 1705);
      // Should use the last one
      testExtractShardId("ABCDEFshard:4324!!Ashard:01220b", 1220);
      // Nothing to extract
      testExtractShardId("ABCDEFsha:222", -1);
      // Regular cases
      testExtractShardId("FC:shard:0", 0);
      testExtractShardId("FC:shard:10", 10);
      testExtractShardId("FC:shard:15", 15);
  */
}

#[test] fn NetAsyncTracingTest_EveryKIteration() {
    todo!();
    /*
      const auto spec = R"DOC(
          name: "example"
          type: "async_scheduling"
          arg {
            name: "enable_tracing"
            i: 1
          }
          arg {
            name: "tracing_mode"
            s: "EVERY_K_ITERATIONS"
          }
          arg {
            name: "tracing_filepath"
            s: "/tmp"
          }
          arg {
            name: "trace_every_nth_batch"
            i: 1
          }
          arg {
            name: "dump_every_nth_batch"
            i: 1
          }
          op {
            output: "out"
            type: "UniformFill"
          }
    )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      Workspace ws;
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      net->Run();
  */
}

#[test] fn NetAsyncTracingTest_GlobalTimeSlice() {
    todo!();
    /*
      const auto spec = R"DOC(
          name: "example"
          type: "async_scheduling"
          arg {
            name: "enable_tracing"
            i: 1
          }
          arg {
            name: "tracing_filepath"
            s: "/tmp"
          }
          arg {
            name: "trace_for_n_ms"
            i: 1
          }
          arg {
            name: "trace_every_n_ms"
            i: 1
          }
          op {
            output: "out"
            type: "UniformFill"
          }
    )DOC";

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));

      Workspace ws;
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      net->Run();
  */
}
