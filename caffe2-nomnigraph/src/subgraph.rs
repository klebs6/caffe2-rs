crate::ix!();

pub type SubgraphType<T,U> = Subgraph<T, U>;
pub type NodeRef<T,U> = *mut NomNode::<T, U>;
pub type EdgeRef<T,U> = *mut Edge::<T, U>;

/**
  | -----------
  | @brief
  | 
  | Effectively a constant reference to
  | a graph.
  | 
  | -----------
  | @note
  | 
  | A Subgraph could actually point to an
  | entire NomGraph.
  | 
  | Subgraphs can only contain references
  | to nodes/edges in a NomGraph.
  | 
  | They are technically mutable, but this
  | should be viewed as a construction helper
  | rather than a fact to be exploited. There
  | are no deleters, for example.
  |
  */
pub struct Subgraph<T,U=EmptyEdgeData> {
    nodes: HashSet<NodeRef<T,U>>,
    edges: HashSet<EdgeRef<T,U>>,
}

impl<T,U> Default for Subgraph<T,U> {
    
    fn default() -> Self {
        todo!();
        /*
            DEBUG_PRINT("Creating instance of Subgraph: %p\n", this)
        */
    }
}

impl<T,U> Subgraph<T,U> {
    
    #[inline] pub fn add_node(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            nodes_.insert(n);
        */
    }
    
    #[inline] pub fn has_node(&self, n: NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            return nodes_.count(n) != 0;
        */
    }
    
    #[inline] pub fn remove_node(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            nodes_.erase(n);
        */
    }
    
    #[inline] pub fn add_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            edges_.insert(e);
        */
    }
    
    #[inline] pub fn has_edge(&self, e: EdgeRef<T,U>) -> bool {
        
        todo!();
        /*
            return edges_.count(e) != 0;
        */
    }
    
    #[inline] pub fn remove_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            edges_.erase(e);
        */
    }
    
    #[inline] pub fn get_nodes(&self) -> &HashSet<NodeRef<T,U>> {
        
        todo!();
        /*
            return nodes_;
        */
    }
    
    #[inline] pub fn get_nodes_count(&self) -> usize {
        
        todo!();
        /*
            return (size_t)nodes_.size();
        */
    }
    
    #[inline] pub fn get_edges(&self) -> &HashSet<EdgeRef<T,U>> {
        
        todo!();
        /*
            return edges_;
        */
    }
    
    #[inline] pub fn print_edges(&mut self)  {
        
        todo!();
        /*
            for (const auto& edge : edges_) {
          printf("Edge: %p (%p -> %p)\n", &edge, edge->tail(), edge->head());
        }
        */
    }
    
    #[inline] pub fn print_nodes(&self)  {
        
        todo!();
        /*
            for (const auto& node : nodes_) {
          printf("NomNode: %p\n", node);
        }
        */
    }
}

