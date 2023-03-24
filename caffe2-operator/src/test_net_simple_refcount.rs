crate::ix!();

#[test] fn net_simple_ref_count_test_correctness() {
    todo!();
    /*
      Workspace ws;
      *(ws.CreateBlob("a")->GetMutable<int32_t>()) = 1;
      NetDef net_def;
      net_def.set_type("simple_refcount");
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"a"}, {"b"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"c"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"b"}, {"d"}));
      net_def.add_op()->CopyFrom(
          CreateOperatorDef("NetSimpleRefCountTest", "", {"c"}, {"e"}));
      // After execution, what should look like is:
      // a = 1
      // b = deallocated
      // c = deallocated
      // d = 3
      // e = 4
      std::unique_ptr<NetBase> net(CreateNet(net_def, &ws));
      net->Run();
      // Note on ASSERT vs EXPECT: ASSERT will quit directly if condition not
      // met, which is why we guard IsType<> calls with ASSERT so that the
      // subsequent Get() calls do not product an exception.
      ASSERT_TRUE(ws.GetBlob("a")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("a")->Get<int32_t>(), 1);
      EXPECT_EQ(ws.GetBlob("b")->GetRaw(), nullptr);
      EXPECT_EQ(ws.GetBlob("c")->GetRaw(), nullptr);
      ASSERT_TRUE(ws.GetBlob("d")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("d")->Get<int32_t>(), 3);
      ASSERT_TRUE(ws.GetBlob("e")->IsType<int32_t>());
      EXPECT_EQ(ws.GetBlob("e")->Get<int32_t>(), 4);
  */
}

