crate::ix!();

#[test] fn subgraph_induce_edges() {
    todo!();
    /*
      auto g = createGraph();
      auto sg = decltype(g)::SubgraphType();
      for (const auto& node : g.getMutableNodes()) {
        sg.addNode(node);
      }

      nom::algorithm::induceEdges(&sg);

      for (const auto& edge : g.getMutableEdges()) {
        EXPECT_TRUE(sg.hasEdge(edge));
      }
  */
}

#[test] fn subgraph_induce_edges_cycle() {
    todo!();
    /*
      auto g = createGraphWithCycle();
      auto sg = decltype(g)::SubgraphType();
      for (const auto& node : g.getMutableNodes()) {
        sg.addNode(node);
      }

      nom::algorithm::induceEdges(&sg);

      for (const auto& edge : g.getMutableEdges()) {
        EXPECT_TRUE(sg.hasEdge(edge));
      }
  */
}
