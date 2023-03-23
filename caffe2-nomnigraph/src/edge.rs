/**
  | This file defines a basic graph API for
  | generic and flexible use with graph
  | algorithms.
  | 
  | Template types:
  | 
  | T : Data stored within a node.
  | 
  | U : Data stored within an edge. When this
  | type is not specified, an empty StorageType
  | is used. If it is specified, only a single
  | type should be given (as supported by
  | the underlying StorageType class).
  |
  */
crate::ix!();


/// \brief Edge within a NomGraph.
pub struct Edge<T,U> {
    base: StorageType<U>,
    tail: NodeRef<T,U>,
    head: NodeRef<T,U>,
}

impl<T,U> Edge<T,U> {

    pub fn new(
        tail: NodeRef<T,U>,
        head: NodeRef<T,U>,
        args: U) -> Self {
    
        todo!();
        /*
            : ::StorageType<U>(std::forward<U>(args)),
            tail_(tail),
            head_(head) 
        DEBUG_PRINT("Creating instance of Edge: %p\n", this);
        */
    }
    
    #[inline] pub fn tail(&self) -> &NodeRef<T,U> {
        
        todo!();
        /*
            return tail_;
        */
    }
    
    #[inline] pub fn head(&self) -> &NodeRef<T,U> {
        
        todo!();
        /*
            return head_;
        */
    }
    
    #[inline] pub fn set_tail(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            tail_ = n;
        */
    }
    
    #[inline] pub fn set_head(&mut self, n: NodeRef<T,U>)  {
        
        todo!();
        /*
            head_ = n;
        */
    }
}

pub struct EmptyEdgeData {}

