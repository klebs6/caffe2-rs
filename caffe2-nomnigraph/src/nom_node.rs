crate::ix!();

/// \brief NomNode within a NomGraph.
pub struct NomNode<T, U=EmptyEdgeData> {

    storage:   StorageType<T>, 
    notifier:  Notifier<NomNode<T, U>>,
    in_edges:  Vec<EdgeRef<T,U>>,
    out_edges: Vec<EdgeRef<T,U>>,
}

impl<T,U> Default for NomNode<T, U> {

    /// \brief Create an empty node.
    fn default() -> Self {
    
        todo!();
        /*
            : ::StorageType<T>()
        */
    }
}

impl<T,U> NomNode<T, U> {
    
    /// \brief Create a node with data.
    pub fn new(data: T) -> Self {
    
        todo!();
        /*
            : ::StorageType<T>(std::move(data)) 
        DEBUG_PRINT("Creating instance of NomNode: %p\n", this);
        */
    }
    
    /**
      | \brief Adds an edge by reference to known
      | in-edges.
      |
      | \p e A reference to an edge that will be
      | added as an in-edge.
      */
    #[inline] pub fn add_in_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            inEdges_.emplace_back(e);
        */
    }

    /**
      | \brief Adds an edge by reference to known
      | out-edges.
      |
      | \p e A reference to an edge that will be
      | added as an out-edge.
      */
    #[inline] pub fn add_out_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            outEdges_.emplace_back(e);
        */
    }

    /**
      | \brief Removes an edge by reference to
      | known in-edges.
      |
      | \p e A reference to an edge that will be
      | removed from in-edges.
      */
    #[inline] pub fn remove_in_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            removeEdgeInternal(inEdges_, e);
        */
    }
    
    /**
      | \brief Removes an edge by reference to
      | known out-edges.
      |
      | \p e A reference to an edge that will be
      | removed from out-edges.
      */
    #[inline] pub fn remove_out_edge(&mut self, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            removeEdgeInternal(outEdges_, e);
        */
    }
    
    #[inline] pub fn get_out_edges(&self) -> &Vec<EdgeRef<T,U>> {
        
        todo!();
        /*
            return outEdges_;
        */
    }
    
    #[inline] pub fn get_in_edges(&self) -> &Vec<EdgeRef<T,U>> {
        
        todo!();
        /*
            return inEdges_;
        */
    }
    
    #[inline] pub fn set_in_edges(&mut self, edges: Vec<EdgeRef<T,U>>)  {
        
        todo!();
        /*
            inEdges_ = std::move(edges);
        */
    }
    
    #[inline] pub fn set_out_edges(&mut self, edges: Vec<EdgeRef<T,U>>)  {
        
        todo!();
        /*
            outEdges_ = std::move(edges);
        */
    }
    
    #[inline] pub fn remove_edge_internal(&mut self, edges: &mut Vec<EdgeRef<T,U>>, e: EdgeRef<T,U>)  {
        
        todo!();
        /*
            auto iter = std::find(edges.begin(), edges.end(), e);
        assert(
            iter != edges.end() &&
            "Attempted to remove edge that isn't connected to this node");
        edges.erase(iter);
        */
    }
}
