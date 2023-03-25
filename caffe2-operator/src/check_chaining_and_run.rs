crate::ix!();

#[inline] pub fn check_chaining_and_run(spec: *const u8, expected: &ExecutionChains)  {
    
    todo!();
    /*
        Workspace ws;
      ws.CreateBlob("in");
      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      {
        net_def.set_num_workers(4);
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
        CHECK_NOTNULL(dag);
        const auto& chains = dag->TEST_execution_chains();
        EXPECT_EQ(chains, expected);
        testExecution(net, net_def.op().size());
      }
    */
}

#[inline] pub fn check_num_chains_and_run(
    spec: *const u8,
    expected_num_chains: i32)  
{
    todo!();
    /*
        Workspace ws;

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      net_def.set_num_workers(4);

      // Create all external inputs
      for (auto inp : net_def.external_input()) {
        ws.CreateBlob(inp);
      }

      {
        std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
        auto* dag = dynamic_cast_if_rtti<AsyncNetBase*>(net.get());
        CHECK_NOTNULL(dag);
        const auto& chains = dag->TEST_execution_chains();
        EXPECT_EQ(expected_num_chains, chains.size());
        testExecution(net, net_def.op().size());
      }
    */
}
