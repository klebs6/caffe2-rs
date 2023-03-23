crate::ix!();

pub type Program = NomGraph<Value>;

///-----------------------------
pub struct ControlFlowGraphImpl<G: GraphType> {
    phantom: PhantomData<G>,

    /*
    // Hack to help debugging in case this class is misused.
    static_assert(
        sizeof(ControlFlowGraphImpl),
        "Template parameter G in "
        "ControlFlowGraph<G> must be of "
        "type Graph<T, U>.");
    */
    /*
       using type = Graph<BasicBlock<T, U>, int>;
       using bbType = BasicBlock<T, U>;
       */
}

impl<G: GraphType> GraphType for ControlFlowGraph<G> {
    type NodeRef = <G as GraphType>::NodeRef;
    type EdgeRef = <G as GraphType>::EdgeRef;

    //TODO: is this the right SubgraphType?
    type SubgraphType = SubgraphType<Self::NodeRef, Self::EdgeRef>;
}

impl<G: GraphType> GraphType for ControlFlowGraphImpl<G> {
    type NodeRef = <G as GraphType>::NodeRef;
    type EdgeRef = <G as GraphType>::EdgeRef;

    //TODO: is this the right SubgraphType?
    type SubgraphType = SubgraphType<Self::NodeRef, Self::EdgeRef>;
}

pub trait BBT {
    type bbType;
}

/**
  | \brief Helper for extracting the type of
  | BasicBlocks given a graph (probably a dataflow
  | graph).  TODO: refactor this to come from
  | something like Graph::NodeDataType
  */
pub type BasicBlockType<G> = <ControlFlowGraphImpl<G> as BBT>::bbType;

pub type BasicBlockRef<G>  = <ControlFlowGraphImpl<G> as GraphType>::NodeRef;

/**
  | \brief Control flow graph is a graph of basic
  | blocks that can be used as an analysis tool.
  |
  | \note G Must be of type Graph<T, U>.
  */
pub struct ControlFlowGraph<G: GraphType> {

    base:      ControlFlowGraphImpl<G>,
    functions: HashMap<String, <ControlFlowGraphImpl<G> as GraphType>::SubgraphType>,
}

impl<G: GraphType> ControlFlowGraph<G> {

    /// Named functions are simply basic blocks stored in labeled Subgraphs
    #[inline] pub fn create_named_function(&mut self, name: String) -> BasicBlockRef<G> {
        
        todo!();
        /*
            assert(name != "anonymous" && "Reserved token anonymous cannot be used");
        auto bb = this->createNode(BasicBlockType<G>());
        assert(functions.count(name) == 0 && "Name already in use.");
        typename ControlFlowGraphImpl<G>::type::SubgraphType sg;
        sg.addNode(bb);
        functions[name] = sg;
        return bb;
        */
    }

    /// Anonymous functions are aggregated into a single Subgraph
    #[inline] pub fn create_anonymous_function(&mut self) -> BasicBlockRef<G> {
        
        todo!();
        /*
            if (!functions.count("anonymous")) {
          functions["anonymous"] =
              typename ControlFlowGraphImpl<G>::type::SubgraphType();
        }

        auto bb = this->createNode(BasicBlockType<G>());
        functions["anonymous"].addNode(bb);
        return bb;
        */
    }
}

