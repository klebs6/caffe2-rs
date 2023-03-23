crate::ix!();

/**
  | -----------
  | @brief
  | 
  | A simple graph implementation
  | 
  | Everything is owned by the graph to simplify
  | storage concerns.
  |
  */
pub struct NomGraph<T,U=EmptyEdgeData> {
    nodes:      LinkedList<NomNode<T,U>>,
    edges:      LinkedList<Edge<T,U>>,
    node_refs:  HashSet<NodeRef<T,U>>,
}

impl<T,U> GraphType for NomGraph<T,U> {
    default type EdgeRef      = EdgeRef<T,U>;
    default type NodeRef      = NodeRef<T,U>;
    default type SubgraphType = SubgraphType<T,U>;
}

impl<T,U> Default for NomGraph<T, U> {
    
    fn default() -> Self {
        todo!();
        /*
            DEBUG_PRINT("Creating instance of NomGraph: %p\n", this)
        */
    }
}

impl<T,U> NomGraph<T, U> {
    
    /**
      | \brief Creates a node and retains
      | ownership of it.
      |
      | \p data An rvalue of the data being held
      | in the node.
      |
      | \return A reference to the node created.
      */
    #[inline] pub fn create_node_with_data(&mut self, data: T) -> NodeRef<T,U> {
        
        todo!();
        /*
            return createNodeInternal(NomNode<T, U>(std::move(data)));
        */
    }
    
    #[inline] pub fn create_node_with_arg<Arg>(&mut self, arg: Arg) -> NodeRef<T,U> {
    
        todo!();
        /*
            return createNode(T(std::forward<Arg>(arg)));
        */
    }
    
    #[inline] pub fn create_node(&mut self) -> NodeRef<T,U> {
        
        todo!();
        /*
            return createNodeInternal(NomNode<T, U>());
        */
    }

    /**
      | Validates the graph.  Returns true if the
      | graph is valid and false if any node or
      | edge referenced in the graph is not
      | actually present in the graph.
      */
    #[inline] pub fn is_valid(&mut self) -> bool {
        
        todo!();
        /*
            for (auto& node : getMutableNodes()) {
          for (auto& inEdge : node->getInEdges()) {
            if (!hasEdge(inEdge)) {
              DEBUG_PRINT("Invalid inEdge %p on node %p\n", inEdge, node);
              return false;
            }
          }
          for (auto& outEdge : node->getOutEdges()) {
            if (!hasEdge(outEdge)) {
              DEBUG_PRINT("invalid outEdge %p on node %p\n", outEdge, node);
              return false;
            }
          }
          // Check validity of nodeRefs_
          if (!hasNode(node)) {
            DEBUG_PRINT("Invalid node %p\n", node);
            return false;
          }
        }
        for (auto& edge : getMutableEdges()) {
          if (!hasNode(edge->tail())) {
            DEBUG_PRINT("Invalid tail on edge %p\n", edge);
            return false;
          }
          if (!hasNode(edge->head())) {
            DEBUG_PRINT("Invalid head on edge %p\n", edge);
            return false;
          }
        }
        return true;
        */
    }

    /**
      | Swap two nodes.
      |
      | Any edge V -> N1 becomes V -> N2, and N1
      | -> V becomes N2 -> V.
      */
    #[inline] pub fn swap_nodes(&mut self, n1: NodeRef<T,U>, n2: NodeRef<T,U>)  {
        
        todo!();
        /*
            // First rectify the edges
        for (auto& inEdge : n1->getInEdges()) {
          inEdge->setHead(n2);
        }
        for (auto& outEdge : n1->getOutEdges()) {
          outEdge->setTail(n2);
        }
        for (auto& inEdge : n2->getInEdges()) {
          inEdge->setHead(n1);
        }
        for (auto& outEdge : n2->getOutEdges()) {
          outEdge->setTail(n1);
        }
        // Then simply copy the edge vectors around
        auto n1InEdges = n1->getInEdges();
        auto n1OutEdges = n1->getOutEdges();
        auto n2InEdges = n2->getInEdges();
        auto n2OutEdges = n2->getOutEdges();

        n1->setOutEdges(n2OutEdges);
        n1->setInEdges(n2InEdges);
        n2->setOutEdges(n1OutEdges);
        n2->setInEdges(n1InEdges);
        */
    }

    /**
      | \brief Replace a node in the graph with
      | another node.
      |
      | \note The node replaced simply has its
      | edges cut, but it not deleted from the
      | graph.  Call NomGraph::deleteNode to
      | delete it.
      |
      | \p oldNode A node to be replaced in the
      | graph.
      |
      | \p newNode The node that inherit the old
      | node's in-edges and out-edges.
      */
    #[inline] pub fn replace_node(&mut self, old_node: &NodeRef<T,U>, new_node: &NodeRef<T,U>)  {
        
        todo!();
        /*
            replaceInEdges(oldNode, newNode);
        replaceOutEdges(oldNode, newNode);
        */
    }

    /**
      | All out-edges oldNode -> V will be replaced
      | with newNode -> V
      |
      */
    #[inline] pub fn replace_out_edges(&mut self, old_node: &NodeRef<T,U>, new_node: &NodeRef<T,U>)  {
        
        todo!();
        /*
            const auto edges = oldNode->getOutEdges();

        for (const auto& edge : edges) {
          edge->setTail(newNode);
          oldNode->removeOutEdge(edge);
          newNode->addOutEdge(edge);
        }
        */
    }

    /**
      | All in-edges V -> oldNode will be replaced
      | with V -> newNode
      |
      */
    #[inline] pub fn replace_in_edges(&mut self, old_node: &NodeRef<T,U>, new_node: &NodeRef<T,U>)  {
        
        todo!();
        /*
            const auto edges = oldNode->getInEdges();

        for (const auto& edge : edges) {
          edge->setHead(newNode);
          oldNode->removeInEdge(edge);
          newNode->addInEdge(edge);
        }
        */
    }

    /**
      | \brief Creates a directed edge and retains
      | ownership of it.
      |
      | \p tail The node that will have this edge
      | as an out-edge.
      |
      | \p head The node that will have this edge
      | as an in-edge.
      |
      | \return A reference to the edge created.
      */
    #[inline] pub fn create_edge(&mut self, 
        tail: NodeRef<T,U>,
        head: NodeRef<T,U>,
        data: U) -> EdgeRef<T,U> {
        
        todo!();
        /*
            DEBUG_PRINT("Creating edge (%p -> %p)\n", tail, head);
        this->edges_.emplace_back(
            Edge<T, U>(tail, head, std::forward<U>(data)));
        EdgeRef<T,U> e = &this->edges_.back();
        head->addInEdge(e);
        tail->addOutEdge(e);
        return e;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Get a reference to the edge between two
      | nodes if it exists. Returns nullptr
      | if the edge does not exist.
      |
      */
    #[inline] pub fn get_edge_if_exists(&self, tail: NodeRef<T,U>, head: NodeRef<T,U>) -> EdgeRef<T,U> {
        
        todo!();
        /*
            for (auto& inEdge : head->getInEdges()) {
          if (inEdge->tail() == tail) {
            return inEdge;
          }
        }
        return nullptr;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Returns true if there is an edge between
      | the given two nodes.
      |
      */
    #[inline] pub fn has_edge_between_given_nodes(&self, tail: NodeRef<T,U>, head: NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            return getEdgeIfExists(tail, head);
        */
    }
    
    #[inline] pub fn has_edge(&self, e: EdgeRef<T,U>) -> bool {
        
        todo!();
        /*
            for (auto& edge : edges_) {
          if (e == &edge) {
            return true;
          }
        }
        return false;
        */
    }

    /**
      | \brief Get a reference to the edge between
      | two nodes if it exists.
      |
      | note: will fail assertion if the edge does
      | not exist.
      */
    #[inline] pub fn get_edge(&self, tail: NodeRef<T,U>, head: NodeRef<T,U>) -> EdgeRef<T,U> {
        
        todo!();
        /*
            auto result = getEdgeIfExists(tail, head);
        assert(result && "Edge doesn't exist.");
        return result;
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Deletes a node from the graph.
      | 
      | -----------
      | @param n
      | 
      | A reference to the node.
      |
      */
    #[inline] pub fn delete_node(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            if (!hasNode(n)) {
          return;
        }

        auto inEdges = n->inEdges_;
        for (auto& edge : inEdges) {
          deleteEdge(edge);
        }
        auto outEdges = n->outEdges_;
        for (auto& edge : outEdges) {
          deleteEdge(edge);
        }

        for (auto i = nodes_.begin(); i != nodes_.end(); ++i) {
          if (&*i == n) {
            nodeRefs_.erase(n);
            nodes_.erase(i);
            break;
          }
        }
        */
    }

    /// Delete all nodes in the set.
    #[inline] pub fn delete_nodes(&mut self, nodes: &HashSet<NodeRef<T,U>>)  {
        
        todo!();
        /*
            for (auto node : nodes) {
          deleteNode(node);
        }
        */
    }
    
    #[inline] pub fn has_node(&self, node: NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            return nodeRefs_.find(node) != nodeRefs_.end();
        */
    }

    /**
      | -----------
      | @brief
      | 
      | Deletes a edge from the graph.
      | 
      | \p e A reference to the edge.
      |
      */
    #[inline] pub fn delete_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            e->tail_->removeOutEdge(e);
        e->head_->removeInEdge(e);
        for (auto i = edges_.begin(); i != edges_.end(); ++i) {
          if (&*i == e) {
            edges_.erase(i);
            break;
          }
        }
        */
    }
    
    #[inline] pub fn get_mutable_nodes(&mut self) -> Vec<NodeRef<T,U>> {
        
        todo!();
        /*
            std::vector<NodeRef<T,U>> result;
        for (auto& n : nodes_) {
          DEBUG_PRINT("Adding node to mutable output (%p)\n", &n);
          result.emplace_back(&n);
        }
        return result;
        */
    }
    
    #[inline] pub fn get_nodes_count(&self) -> usize {
        
        todo!();
        /*
            return (size_t)nodes_.size();
        */
    }
    
    #[inline] pub fn get_mutable_edges(&mut self) -> Vec<EdgeRef<T,U>> {
        
        todo!();
        /*
            std::vector<EdgeRef<T,U>> result;
        for (auto& e : edges_) {
          DEBUG_PRINT("Adding edge to mutable output (%p)\n", &e);
          result.emplace_back(&e);
        }
        return result;
        */
    }
    
    #[inline] pub fn get_edges_count(&self) -> usize {
        
        todo!();
        /*
            return (size_t)edges_.size();
        */
    }
    
    #[inline] pub fn print_edges(&mut self)  {
        
        todo!();
        /*
            for (const auto& edge : edges_) {
          printf("Edge: %p (%p -> %p)\n", &edge, edge.tail(), edge.head());
        }
        */
    }
    
    #[inline] pub fn print_nodes(&self)  {
        
        todo!();
        /*
            for (const auto& node : nodes_) {
          printf("NomNode: %p\n", &node);
        }
        */
    }

    /**
      | Note:
      |
      | The move functions below are unsafe.  Use
      | them with caution and be sure to call
      | isValid() after each use.
      */

    /**
      | Move a node from this graph to the destGraph
      |
      */
    #[inline] pub unsafe fn move_node(&mut self, node: NodeRef<T,U>, dest_graph: *mut NomGraph<T,U>)  {
        
        todo!();
        /*
            assert(hasNode(node));
        for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
          if (&(*it) == node) {
            std::list<NomNode<T, U>>& destNodes = destGraph->nodes_;
            destNodes.splice(destNodes.end(), nodes_, it);
            nodeRefs_.erase(node);
            destGraph->nodeRefs_.insert(node);
            break;
          }
        }
        */
    }

    /**
      | Move an edge from this graph to the destGraph
      |
      */
    #[inline] pub unsafe fn move_edge(&mut self, edge: EdgeRef<T,U>, dest_graph: *mut NomGraph<T,U>)  {
        
        todo!();
        /*
            assert(hasEdge(edge));
        assert(destGraph->hasNode(edge->tail()));
        assert(destGraph->hasNode(edge->head()));
        std::list<Edge<T, U>>& destEdges = destGraph->edges_;
        for (auto it = edges_.begin(); it != edges_.end(); ++it) {
          if (&(*it) == edge) {
            destEdges.splice(destEdges.end(), edges_, it);
            break;
          }
        }
        */
    }

    /**
      | Move entire subgraph to destGraph.
      |
      | Be sure to delete in/out edges from this
      | graph first.
      */
    #[inline] pub unsafe fn move_subgraph(&mut self, subgraph: &Subgraph<T,U>, dest_graph: *mut NomGraph<T,U>)  {
        
        todo!();
        /*
            auto sg = subgraph; // Copy to check that all nodes and edges are matched
        std::list<Edge<T, U>>& destEdges = destGraph->edges_;
        for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
          auto node = &(*it);
          if (sg.hasNode(node)) {
            std::list<NomNode<T, U>>& destNodes = destGraph->nodes_;
            destNodes.splice(destNodes.end(), nodes_, it--);
            nodeRefs_.erase(node);
            destGraph->nodeRefs_.insert(node);
            sg.removeNode(node);
          }
        }
        for (auto it = edges_.begin(); it != edges_.end(); ++it) {
          auto edge = &(*it);
          if (sg.hasEdge(edge)) {
            assert(destGraph->hasNode(edge->tail()));
            assert(destGraph->hasNode(edge->head()));
            destEdges.splice(destEdges.end(), edges_, it--);
            sg.removeEdge(edge);
          }
        }
        assert(sg.getNodes().size() == 0);
        assert(sg.getEdges().size() == 0);
        */
    }
    
    #[inline] pub unsafe fn create_node_internal(&mut self, node: NomNode<T,U>) -> NodeRef<T,U> {
        
        todo!();
        /*
            nodes_.emplace_back(std::move(node));
        NodeRef<T,U> nodeRef = &nodes_.back();
        DEBUG_PRINT("Creating node (%p)\n", nodeRef);
        nodeRefs_.insert(nodeRef);
        return nodeRef;
        */
    }
}
