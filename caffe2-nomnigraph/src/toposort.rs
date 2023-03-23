crate::ix!();

/**
  | \brief A function wrapper to infer the graph
  | template parameters.
  |
  | TODO change this to const GraphT& g
  */
#[inline] pub fn topo_sort<G: GraphType>(g: *mut G) -> TopoSortResult<G> {

    todo!();
    /*
        TopoSort<GraphT> t(g);
      return t.run();
    */
}

pub enum TopoSortResultStatus { OK, CYCLE }

pub struct TopoSortResult<G: GraphType> {
    status: TopoSortResultStatus,
    nodes:  Vec<NodeRefT<G>>,
}

pub type NodeRefT<G: GraphType> = <G as GraphType>::NodeRef;

/**
  | \brief Topological sort using DFS.
  |
  | This algorithm takes a Graph object and
  | returns node references in topological order.
  */
pub struct TopoSort<G: GraphType> {
    graph: *mut G,
}

impl<G: GraphType> TopoSort<G> {

    /**
      | \brief performs DFS from given node.
      |
      |  Each node and edge is visited no more
      |  than once.
      |
      |  Visited nodes are pushed into result
      |  vector after all children has been
      |  processed. Return true if cycle is
      |  detected, otherwise false.
      */
    #[inline] pub fn dfs(&mut self, 
        node:   NodeRefT<G>,
        status: &mut HashMap<NodeRefT<G>,i32>,
        nodes:  &mut Vec<NodeRefT<G>>) -> bool {
        
        todo!();
        /*
            // mark as visiting
        status[node] = 1;
        for (const auto& outEdge : node->getOutEdges()) {
          auto& newNode = outEdge->head();
          int newStatus = status[newNode];
          if (newStatus == 0) {
            if (dfs(newNode, status, nodes)) {
              return true;
            }
          } else if (newStatus == 1) {
            // find a node being visited, cycle detected
            return true;
          }
          // ignore visited node
        }
        nodes.push_back(node);
        // mark as visited
        status[node] = 2;
        return false;
        */
    }
    
    pub fn new(graph: *mut G) -> Self {
    
        todo!();
        /*
            : graph(graph)
        */
    }
    
    #[inline] pub fn run(&mut self)  {
        
        todo!();
        /*
            std::vector<NodeRefT<G>> nodes;
        std::unordered_map<NodeRefT<G>, int> status;
        for (auto& node : graph->getMutableNodes()) {
          if (!status[node]) {
            if (dfs(node, status, nodes)) {
              return {Result::CYCLE, {}};
            }
          }
        }
        std::reverse(nodes.begin(), nodes.end());
        return {Result::OK, nodes};
        */
    }
}
