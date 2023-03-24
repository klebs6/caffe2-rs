crate::ix!();

#[test] fn dag_util_test_empty() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test0"
        type: "async_scheduling"
        )DOC";
      Workspace ws;
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      EXPECT_TRUE(chains.empty());
  */
}

/// 4 sync ops forming a diamond
///
#[test] fn dag_util_test_all_sync() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test1"
        type: "async_scheduling"
        external_input: "in"
        op {
          input: "in"
          output: "n1"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          output: "n2"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          output: "n3"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n2"
          input: "n3"
          output: "out"
          type: "DagUtilTestDummySync"
        }
        )DOC";
      Workspace ws;
      ws.CreateBlob("in");
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      dag_utils::ExecutionChains expected{{0, {0, 1, 2, 3}}};
      EXPECT_EQ(chains, expected);
  */
}

/// 3 async ops forming an L shape
///
#[test] fn dag_util_test_all_async() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test2"
        type: "async_scheduling"
        external_input: "in0"
        external_input: "in1"
        op {
          input: "in0"
          output: "n1"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "in1"
          output: "n2"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n1"
          output: "n3"
          type: "DagUtilTestDummyAsync"
        }
        )DOC";
      Workspace ws;
      ws.CreateBlob("in0");
      ws.CreateBlob("in1");
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      dag_utils::ExecutionChains expected{{0, {0}}, {1, {1}}, {2, {2}}};
      EXPECT_EQ(chains, expected);
  */
}

/// 3 sync ops and 1 async op (#2) forming a diamond
///
#[test] fn dag_util_test_mixed0() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test3"
        type: "async_scheduling"
        external_input: "in"
        op {
          input: "in"
          output: "n1"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          output: "n2"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          output: "n3"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n2"
          input: "n3"
          output: "out"
          type: "DagUtilTestDummySync"
        }
        )DOC";
      Workspace ws;
      ws.CreateBlob("in");
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      dag_utils::ExecutionChains expected{{0, {0, 1}}, {2, {2}}, {3, {3}}};
      EXPECT_EQ(chains, expected);
  */
}

/// 3 sync ops and 1 async op (#2) forming a Y shape
///
#[test] fn dag_util_test_mixed1() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test3"
        type: "async_scheduling"
        external_input: "in0"
        external_input: "in1"
        op {
          input: "in0"
          output: "n1"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "in1"
          output: "n2"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          input: "n2"
          output: "n3"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n3"
          output: "out"
          type: "DagUtilTestDummySync"
        }
        )DOC";
      Workspace ws;
      ws.CreateBlob("in0");
      ws.CreateBlob("in1");
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      dag_utils::ExecutionChains expected{{0, {0, 1}}, {2, {2}}, {3, {3}}};
      EXPECT_EQ(chains, expected);
  */
}

/**
  | More complicated mixed case. * means async
  |
  |  0* -> 1* -> 2
  |    |
  |  3 -> 4 -> 5
  |  |  |
  |  |    6
  |   - -> 8*
  |  7* -/
  */
#[test] fn dag_util_test_mixed2() {
    todo!();
    /*
      const auto spec = R"DOC(
        name: "test4"
        type: "async_scheduling"
        external_input: "in0"
        external_input: "in1"
        external_input: "in2"
        op {
          input: "in0"
          output: "n1"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n1"
          output: "n2"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n2"
          output: "out0"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "in1"
          output: "n3"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n1"
          input: "n3"
          output: "n4"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n4"
          output: "out1"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "n3"
          output: "out2"
          type: "DagUtilTestDummySync"
        }
        op {
          input: "in2"
          output: "n7"
          type: "DagUtilTestDummyAsync"
        }
        op {
          input: "n3"
          input: "n7"
          output: "out3"
          type: "DagUtilTestDummyAsync"
        }
        )DOC";
      Workspace ws;
      ws.CreateBlob("in0");
      ws.CreateBlob("in1");
      ws.CreateBlob("in2");
      DagUtilTestContext t(spec, &ws);
      auto chains = t.computeChains();
      dag_utils::ExecutionChains expected{
          {0, {0}}, {1, {1}}, {3, {3, 6}}, {4, {4, 2, 5}}, {7, {7}}, {8, {8}}};
      EXPECT_EQ(chains, expected);
  */
}
