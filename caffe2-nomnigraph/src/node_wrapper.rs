crate::ix!();

pub type EdgeWrapper<T,U>     = <GraphWrapper::<T, U> as HasEdgeWrapper>::EdgeWrapper;
pub type WrappedGraph<T,U>    = NomGraph::<NodeWrapper<T,U>, EdgeWrapper<T,U>>;
pub type WrappedSubgraph<T,U> = Subgraph::<NodeWrapper<T,U>, EdgeWrapper<T,U>>;

impl<T,U> GraphType for WrappedGraph<T,U> {
    type NodeRef = NodeRef<T,U>;
    type EdgeRef = EdgeRef<T,U>;
    type SubgraphType = WrappedSubgraph<T,U>;
}

pub struct NodeWrapper<T,U> {
    node:     NodeRef<T,U>,
    index:    i32, // default = -1
    low_link: i32, // default = -1
    on_stack: bool, // default = false
}

impl<T,U> NodeWrapper<T,U> {
    
    pub fn new(n: NodeRef<T,U>) -> Self {
    
        todo!();
        /*
            : node(n)
        */
    }
}

