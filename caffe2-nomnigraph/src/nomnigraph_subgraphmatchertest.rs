crate::ix!();

use crate::{
    GraphType,
    Graph,
    MatchGraph,
    NomGraph,
    MatchPredicate
};

pub type NodeType           = String;
pub type Criteria           = String;
pub type TestGraph          = NomGraph<NodeType>;
pub type TestMatchGraph     = MatchGraph<TestGraph>;
pub type TestMatchPredicate = MatchPredicate<TestGraph>;

lazy_static!{

    /*
    /// Have just one TestMatchGraph in the tests to make it less verbose to create
    /// the match graphs.
    static ref graph: Arc<Mutex<TestMatchGraph>> = Arc::new(Mutex::new(TestMatchGraph::default()));
    */
}

/// Call reset before creating a new TestMatchGraph.
#[inline] pub fn reset()  {
    
    todo!();
    /*
        graph = TestMatchGraph();
    */
}

/**
  | Node matches a criteria (string) if
  | the data string is the same as the criteria.
  | Special case: "*" will match any thing.
  |
  */
#[inline] pub fn test_match_predicate(criteria: &Criteria) -> TestMatchPredicate {
    
    todo!();
    /*
        return TestMatchPredicate([criteria](TestGraph::NodeRef node) {
        return criteria == "*" || criteria == node->data();
      });
    */
}

#[inline] pub fn any() -> Criteria {
    
    todo!();
    /*
        return Criteria("*");
    */
}

/**
  | Helper methods to make it less verbose
  | to create match graphs.
  |
  */
#[inline] pub fn tree(
    root:     &Criteria,
    children: &Vec<<TestMatchGraph as GraphType>::NodeRef>,
    count:    Option<i32>) -> <TestMatchGraph as GraphType>::NodeRef {
    let count: i32 = count.unwrap_or(1);

    todo!();
    /*
        auto result =
          graph.createNode(std::move(testMatchPredicate(root).count(count)));
      for (auto& child : children) {
        graph.createEdge(result, child);
      }
      return result;
    */
}

#[inline] pub fn non_terminal(
    root: &Criteria, 
    count: Option<i32>) -> <TestMatchGraph as GraphType>::NodeRef {
    let count: i32 = count.unwrap_or(1);

    todo!();
    /*
        return graph.createNode(
          std::move(testMatchPredicate(root).count(count).nonTerminal()));
    */
}

#[inline] pub fn test_graph_node_printer(node: <TestGraph as GraphType>::NodeRef) -> HashMap<String,String> {
    
    todo!();
    /*
        std::map<std::string, std::string> labelMap;
      labelMap["label"] = node->data();
      return labelMap;
    */
}

pub type TestGraph_NodeRef = <TestGraph as GraphType>::NodeRef;

/**
  | Attempts to create a realistic dataflow
  | graph that shows a fuse procedure.
  |
  */
pub struct DataFlowTestGraph {
    num_inputs:  i32, // default = 4
    graph:       TestGraph,
    opb:         TestGraph_NodeRef,
    opf:         TestGraph_NodeRef,
    opc:         TestGraph_NodeRef,
    opg:         TestGraph_NodeRef,
    data_out:    TestGraph_NodeRef,
}

/**
 |   Realistic data flow test graph.
 |
 |
 |                         +---------------+
 |                         |               |
 |                         |  +---------+  |  +---------+
 |   +---------------------+  | input_A |  |  | input_B |
 |   |                        +---------+  |  +---------+
 |   |                          |          |    |
 |   |                          |          |    |
 |   |                          v          v    v
 | +---------++---------+     +-------------------------+     +--------+
 | | input_C || input_D | --> |           opC           | --> | dataC2 |
 | +---------++---------+     +-------------------------+     +--------+
 |                              |
 |                              |
 |                              v
 |                            +---------+
 |                            |  dataC  | -+
 |                            +---------+  |
 |                              |          |
 |                              |          |
 |                              v          |
 |                            +---------+  |
 |                            |   opB   | <+
 |                            +---------+
 |                              |
 |                              |
 |                              v
 |                            +---------+
 |                            |  dataB  |
 |                            +---------+
 |                              |
 |                              |
 |                              v
 |                            +---------+
 |                            |   opF   |
 |                            +---------+
 |                              |
 |                              |
 |                              v
 |                            +---------+
 |                            |  dataF  |
 |                            +---------+
 |                              |
 |                              |
 |                              v
 |            +---------+     +---------+
 |            |  dataI  | --> |   opG   |
 |            +---------+     +---------+
 |                              |
 |                              |
 |                              v
 |                            +---------+
 |                            | dataOut |
 |                            +---------+
 */
impl Default for DataFlowTestGraph {

    fn default() -> Self {
        todo!();
        /*
            opC = graph.createNode("opC");

        for (int i = 0; i < numInputs; i++) {
          auto dataInput = graph.createNode("input");
          graph.createEdge(dataInput, opC);
        }

        auto dataC = graph.createNode("dataC");
        auto dataC2 = graph.createNode("dataC2");
        graph.createEdge(opC, dataC);
        graph.createEdge(opC, dataC2);

        opB = graph.createNode("opB");
        // There are 2 edges
        graph.createEdge(dataC, opB);
        graph.createEdge(dataC, opB);

        auto dataB = graph.createNode("dataB");
        graph.createEdge(opB, dataB);

        opF = graph.createNode("opF");
        graph.createEdge(dataB, opF);

        auto dataF = graph.createNode("dataF");
        graph.createEdge(opF, dataF);

        auto dataI = graph.createNode("dataI");

        opG = graph.createNode("opG");
        graph.createEdge(dataF, opG);
        graph.createEdge(dataI, opG);

        dataOut = graph.createNode("dataOut");
        graph.createEdge(opG, dataOut);

        // Use nom::converters::convertToDotString(&graph, TestGraphNodePrinter)
        // to visualize the graph
        */
    }
}

///------------------------------
struct DataFlowTestGraphCriteria {
    match_opc_output:  <TestMatchGraph as GraphType>::NodeRef,
    match_opg:         <TestMatchGraph as GraphType>::NodeRef,
}

impl Default for DataFlowTestGraphCriteria {
    
    fn default() -> Self {
        todo!();
        /*
            auto matchOpCInputs =
            graph.createNode(std::move(testMatchPredicate(Criteria("input"))
                                           .starCount()
                                           .nonTerminal()
                                           .excludeFromSubgraph()));
        auto matchOpC = graph.createNode(testMatchPredicate("opC"));
        graph.createEdge(matchOpCInputs, matchOpC);

        matchOpCOutput = graph.createNode(testMatchPredicate(any()));
        graph.createEdge(matchOpC, matchOpCOutput);

        auto matchOpB = graph.createNode(testMatchPredicate("opB"));
        graph.createEdge(matchOpCOutput, matchOpB);
        graph.createEdge(matchOpCOutput, matchOpB);

        auto matchOpBOutput = graph.createNode(testMatchPredicate(any()));
        graph.createEdge(matchOpB, matchOpBOutput);

        auto matchOpF = graph.createNode(testMatchPredicate("opF"));
        graph.createEdge(matchOpBOutput, matchOpF);

        auto matchOpFOutput = graph.createNode(testMatchPredicate(any()));
        graph.createEdge(matchOpF, matchOpFOutput);

        matchOpG = graph.createNode(testMatchPredicate("opG"));
        auto matchDataI = graph.createNode(std::move(
            testMatchPredicate(any()).nonTerminal().excludeFromSubgraph()));
        graph.createEdge(matchOpFOutput, matchOpG);
        graph.createEdge(matchDataI, matchOpG)
        */
    }
}

#[inline] pub fn get_in_node(
    node: <TestGraph as GraphType>::NodeRef, 
    index: i32) -> <TestGraph as GraphType>::NodeRef 
{
    todo!();
    /*
        return node->getInEdges()[index]->tail();
    */
}

#[inline] pub fn is_subgraph_match(
    node_ref:               <TestGraph as GraphType>::NodeRef,
    criteria:               &<TestMatchGraph as GraphType>::NodeRef,
    invert_graph_traversal: Option<bool>) -> bool {

    let invert_graph_traversal: bool = invert_graph_traversal.unwrap_or(true);

    todo!();
    /*
        return graph.isSubgraphMatch(nodeRef, criteria, invertGraphTraversal)
          .isMatch();
    */
}

/// Simple test cases for node matching criteria.
#[test] fn subgraph_matcher_is_node_match() {
    todo!();
    /*
      TestGraph g;
      auto n1 = g.createNode("Hello");
      auto n2 = g.createNode("Le");
      g.createEdge(n1, n2);

      EXPECT_TRUE(graph.isNodeMatch(n1, testMatchPredicate("Hello")));
      EXPECT_FALSE(graph.isNodeMatch(n1, testMatchPredicate("G")));
      EXPECT_TRUE(graph.isNodeMatch(n2, testMatchPredicate("Le")));
      EXPECT_FALSE(graph.isNodeMatch(n2, testMatchPredicate("le")));
  */
}


/// Test subtree matching with a simple tree graph.
#[test] fn subgraph_matcher_is_subtree_match() {
    todo!();
    /*
      TestGraph graph;
      auto n1 = graph.createNode("1");
      auto n2 = graph.createNode("2");
      auto n3 = graph.createNode("3");
      auto n4 = graph.createNode("4");
      auto n5 = graph.createNode("5");
      auto n6 = graph.createNode("6");
      auto n7 = graph.createNode("7");

      graph.createEdge(n1, n2);
      graph.createEdge(n2, n3);
      graph.createEdge(n2, n4);
      graph.createEdge(n1, n5);
      graph.createEdge(n5, n6);
      graph.createEdge(n5, n7);
      /*       N1
             /     \
          N2         N5
        /    \     /    \
      N3     N4   N6   N7
      */

      reset();
      auto subtree = Tree(any(), {Tree(any()), Tree(any())});
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n4, subtree, false));

      EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
      EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));

      reset();
      subtree = Tree(Criteria("5"), {Tree(any()), Tree(any())});
      EXPECT_FALSE(isSubgraphMatch(n2, subtree, false));
      EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));

      reset();
      subtree = Tree(any(), {Tree(any()), Tree(Criteria("4"))});
      EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n5, subtree, false));

      reset();
      // Accepts non terminal node
      subtree = Tree(any(), {NonTerminal(any()), NonTerminal(any())});
      EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));
      EXPECT_TRUE(isSubgraphMatch(n2, subtree, false));
      EXPECT_TRUE(isSubgraphMatch(n5, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n3, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n4, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n6, subtree, false));
      EXPECT_FALSE(isSubgraphMatch(n7, subtree, false));
  */
}

/**
  | Test subtree matching in which * (repeated)
  | matching of children is allowed.
  |
  */
#[test] fn subgraph_matcher_is_subtree_match_repeated() {
    todo!();
    /*
      TestGraph graph;
      auto n1 = graph.createNode("1");
      auto n2 = graph.createNode("2");
      auto n3A = graph.createNode("3");
      auto n3B = graph.createNode("3");
      auto n4 = graph.createNode("4");
      auto n5A = graph.createNode("5");
      auto n5B = graph.createNode("5");
      auto n5C = graph.createNode("5");
      graph.createEdge(n1, n2);
      graph.createEdge(n1, n3A);
      graph.createEdge(n1, n3B);
      graph.createEdge(n1, n4);
      graph.createEdge(n1, n4);
      graph.createEdge(n1, n5A);
      graph.createEdge(n1, n5B);
      graph.createEdge(n1, n5C);

      reset();
      auto subtree = Tree(any(), {Tree(Criteria("2"))});
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree =
          Tree(any(), {Tree(Criteria("2"), {}, TestMatchPredicate::kStarCount)});
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

      reset();
      // clang-format off
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, 2),
        Tree(Criteria("4"), {}, 2),
        Tree(Criteria("5"), {}, 3)
      });
      EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, 2),
        Tree(Criteria("4"), {}, 2),
        Tree(Criteria("5"), {}, 4)
      });
      // Failes because exepected 4 matches of n5 but found 3.
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, 2),
        Tree(Criteria("4"), {}, 2),
        Tree(Criteria("5"), {}, TestMatchPredicate::kStarCount)
      });
      EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, TestMatchPredicate::kStarCount),
        Tree(Criteria("4"), {}, 2),
        Tree(Criteria("5"), {}, TestMatchPredicate::kStarCount)
      });
      EXPECT_TRUE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, TestMatchPredicate::kStarCount),
      });
      // Fails because there are unmatched edges.
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));

      reset();
      subtree = Tree(any(), {
        Tree(Criteria("2")),
        Tree(Criteria("3"), {}, 2),
        Tree(Criteria("4")),
        Tree(Criteria("5"), {}, 3)
      });
      // Fails because the count is wrong; we have 2 edges to node N4 while
      // the pattern expects only 1.
      EXPECT_FALSE(isSubgraphMatch(n1, subtree, false));
      // clang-format on
  */
}


#[test] fn subgraph_matcher_dag_matching() {
    todo!();
    /*
      reset();

      // clang-format off
      auto n4match = Tree(Criteria("4"), {
        Tree(Criteria("5"))
      });
      auto subgraph = Tree(Criteria("1"), {
        Tree(Criteria("2"), {
          n4match
        }),
        Tree(Criteria("3"), {
          n4match
        }),
      });
      // clang-format on

      {
        TestGraph graph;
        auto n1 = graph.createNode("1");
        auto n2 = graph.createNode("2");
        auto n3 = graph.createNode("3");
        auto n4 = graph.createNode("4");
        auto n5 = graph.createNode("5");

        graph.createEdge(n1, n2);
        graph.createEdge(n1, n3);
        graph.createEdge(n2, n4);
        graph.createEdge(n3, n4);
        graph.createEdge(n4, n5);

        /*       N1
               /     \
            N2         N3
                \   /
                 N4
                 |
                 N5
        */

        EXPECT_TRUE(isSubgraphMatch(n1, subgraph, false));
      }

      {
        TestGraph graph;
        auto n1 = graph.createNode("1");
        auto n2 = graph.createNode("2");
        auto n3 = graph.createNode("3");
        auto n4A = graph.createNode("4");
        auto n4B = graph.createNode("4");
        auto n5 = graph.createNode("5");

        graph.createEdge(n1, n2);
        graph.createEdge(n1, n3);
        graph.createEdge(n2, n4A);
        graph.createEdge(n3, n4B);
        graph.createEdge(n4A, n5);
        graph.createEdge(n4B, n5);

        /*       N1
               /    \
            N2       N3
            /          \
           N4A        N4B
              \     /
                N5
        */

        // This should fail because n4A and n4B are not the same node.
        EXPECT_FALSE(isSubgraphMatch(n1, subgraph, false));
      }
  */
}


#[test] fn subgraph_matcher_dag_matching_multi_edges() {
    todo!();
    /*
      reset();

      // clang-format off
      auto n2match = Tree(Criteria("2"));
      auto subgraph = Tree(Criteria("1"), {
        n2match,
        n2match
      });
      // clang-format on

      {
        TestGraph graph;
        auto n1 = graph.createNode("1");
        auto n2 = graph.createNode("2");

        graph.createEdge(n1, n2);
        graph.createEdge(n1, n2);

        EXPECT_TRUE(isSubgraphMatch(n1, subgraph, false));
      }

      {
        TestGraph graph;
        auto n1 = graph.createNode("1");
        auto n2A = graph.createNode("2");
        auto n2B = graph.createNode("2");

        graph.createEdge(n1, n2A);
        graph.createEdge(n1, n2B);

        EXPECT_FALSE(isSubgraphMatch(n1, subgraph, false));
      }
  */
}


#[test] fn subgraph_matcher_dag_matching_random_large_graph() {
    todo!();
    /*
      reset();
      // clang-format off
      auto n4match = Tree(any(), {
        NonTerminal(any(), 1)
      });
      auto subtree = Tree(any(), {
        Tree(any(), {
          n4match
        }),
        Tree(any(), {
          n4match
        }),
      });
      // clang-format on
      /*       N1
             /     \
          N2         N3
              \   /
               N4
               |
               N5
      */

      // Look for the diamond pattern in a random large graph.
      TestGraph graph;
      std::vector<nom::Graph<std::string>::NodeRef> nodes;

      // Here we create a test graph and then randomly embed the above
      // pattern into the graph repeatedly (numPatterns times).
      // The actual number of match will be less than numPatterns because the
      // embedded patterns can overlap which become unmatched subgraphs.
      const int numNodes = 50000;
      const int numPatterns = 5000;

      for (int i = 0; i < numNodes; i++) {
        auto node = graph.createNode("Node");
        nodes.emplace_back(node);
      }

      TestRandom random(517);
      for (int i = 0; i < numPatterns; i++) {
        std::vector<int> nodeIdx;
        for (int k = 0; k < 5; k++) {
          nodeIdx.emplace_back(random.nextInt() % numNodes);
        }
        graph.createEdge(nodes[nodeIdx[0]], nodes[nodeIdx[1]]);
        graph.createEdge(nodes[nodeIdx[0]], nodes[nodeIdx[2]]);
        graph.createEdge(nodes[nodeIdx[1]], nodes[nodeIdx[3]]);
        graph.createEdge(nodes[nodeIdx[2]], nodes[nodeIdx[3]]);
        graph.createEdge(nodes[nodeIdx[3]], nodes[nodeIdx[4]]);
      }
      EXPECT_EQ(graph.getEdgesCount(), 5 * numPatterns);

      int countMatch = 0;
      for (auto node : graph.getMutableNodes()) {
        if (isSubgraphMatch(node, subtree, false)) {
          countMatch++;
        }
      }
      EXPECT_EQ(countMatch, 1072);
  */
}


#[test] fn subgraph_matcher_is_subtree_match_realistic() {
    todo!();
    /*
      reset();
      auto graph = DataFlowTestGraph();
      auto subtree = DataFlowTestGraphCriteria().matchOpG;

      EXPECT_FALSE(isSubgraphMatch(graph.opF, subtree));
      EXPECT_FALSE(isSubgraphMatch(graph.opC, subtree));
      EXPECT_FALSE(isSubgraphMatch(graph.opB, subtree));
      EXPECT_FALSE(isSubgraphMatch(graph.dataOut, subtree));

      EXPECT_TRUE(isSubgraphMatch(graph.opG, subtree));
  */
}


#[test] fn subgraph_matcher_replace_graph_realistic() {
    todo!();
    /*
      reset();
      auto testGraph = DataFlowTestGraph();
      auto subtree = DataFlowTestGraphCriteria();

      graph.replaceSubgraph(
          testGraph.graph,
          subtree.matchOpG,
          [subtree](
              TestGraph& g,
              TestGraph::NodeRef opG,
              const TestMatchGraph::SubgraphMatchResultType& matchResult) {
            auto fusedNode = g.createNode("opFused");
            auto opC = getInNode(
                matchResult.getMatchNodeMap()->at(subtree.matchOpCOutput), 0);
            g.replaceOutEdges(opG, fusedNode);
            g.replaceInEdges(opG, fusedNode);
            g.replaceInEdges(opC, fusedNode);
            g.deleteNodes(matchResult.getMatchedSubgraph()->getNodes());
            return true;
          });

      // Now the nodes are:
      // - NumInputs input nodes
      // - dataI node
      // - fused node
      // - output node
      // - dataC2 node
      auto nodes = testGraph.graph.getMutableNodes();

      // Test that the graph is transformed as expected.
      EXPECT_EQ(nodes.size(), testGraph.numInputs + 4);
      TestGraph::NodeRef opFused;
      TestGraph::NodeRef dataI;
      TestGraph::NodeRef dataOut;
      for (auto node : nodes) {
        if (node->data() == "opFused") {
          opFused = node;
        } else if (node->data() == "dataOut") {
          dataOut = node;
        } else if (node->data() == "dataI") {
          dataI = node;
        }
      }

      EXPECT_EQ(getInNode(dataOut, 0), opFused);

      EXPECT_EQ(opFused->getInEdges().size(), testGraph.numInputs + 1);
      EXPECT_EQ(getInNode(opFused, 0), dataI);
      for (int i = 1; i <= testGraph.numInputs; i++) {
        EXPECT_EQ(getInNode(opFused, i)->data(), "input");
      }

      // Use nom::converters::convertToDotString(&graph.graph, TestGraphNodePrinter)
      // to visualize. The transformed graph looks like This
      /*

                    +---------++---------+
                    | input_A || input_D |
                    +---------++---------+
                      |          |
                      |          |
                      v          v
    +---------+     +--------------------+     +---------+
    | input_B | --> |      opFused       | <-- | input_C |
    +---------+     +--------------------+     +---------+
                      |          ^
                      |          |
                      v          |
                    +---------++---------+
                    | dataOut ||  dataI  |
                    +---------++---------+
       */
  */
}

