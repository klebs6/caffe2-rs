crate::ix!();


/**
  | \brief A basic block holds a reference to
  | a subgraph of the data flow graph as well as
  | an ordering on instruction execution.  Basic
  | blocks are used for control flow analysis.
  */
pub struct BasicBlock<T,U> {
    
    nodes:         Subgraph<T,U>,
    instructions:  Vec<NodeRef<T,U>>,

    /*
      | Because we reference a dataflow graph,
      | we need to register callbacks for when
      | the dataflow graph is modified.
      |
      */
    callbacks:     HashMap<NodeRef<T,U>, *mut NotifierCallback<Node>>,
}

impl<T,U> Drop for BasicBlock<T,U> {
    fn drop(&mut self) {
        todo!();
        /* 
        for (auto pair : callbacks_) {
          pair.first->deleteDestructorCallback(pair.second);
        }
       */
    }
}

impl<T,U> BasicBlock<T, U> {
    
    #[inline] pub fn track_node(&mut self, node: NodeRef<T,U>)  {
        
        todo!();
        /*
            callbacks_[node] = node->registerDestructorCallback([&](NodeRef n) {
          assert(
              hasInstruction(n) &&
              "Destructor callback invoked on untracked node in BasicBlock.");
          deleteInstruction(n);
        });
        nodes_.addNode(node);
        */
    }
    
    #[inline] pub fn untrack_node(&mut self, node: NodeRef<T,U>)  {
        
        todo!();
        /*
            callbacks_.erase(node);
        nodes_.removeNode(node);
        */
    }
    
    #[inline] pub fn push_instruction_node(&mut self, node: NodeRef<T,U>)  {
        
        todo!();
        /*
            assert(
            isa<Instruction>(node->data()) &&
            "Cannot push non-instruction node to basic block.");
        instructions_.emplace_back(node);
        trackNode(node);
        */
    }
    
    #[inline] pub fn get_instructions(&self) -> &Vec<NodeRef<T,U>> {
        
        todo!();
        /*
            return instructions_;
        */
    }
    
    #[inline] pub fn get_mutable_instructions(&mut self) -> *mut Vec<NodeRef<T,U>> {
        
        todo!();
        /*
            return &instructions_;
        */
    }
    
    #[inline] pub fn has_instruction(&self, instr: NodeRef<T,U>) -> bool {
        
        todo!();
        /*
            return nodes_.hasNode(instr);
        */
    }
    
    #[inline] pub fn insert_instruction_before(&mut self, new_instr: NodeRef<T,U>, instr: NodeRef<T,U>)  {
        
        todo!();
        /*
            auto it =
            std::find(std::begin(instructions_), std::end(instructions_), instr);
        instructions_.insert(it, newInstr);
        trackNode(newInstr);
        */
    }
    
    #[inline] pub fn move_instruction_before(&mut self, instr1: NodeRef<T,U>, instr2: NodeRef<T,U>)  {
        
        todo!();
        /*
            assert(hasInstruction(instr1) && "Instruction not in basic block.");
        assert(hasInstruction(instr2) && "Instruction not in basic block.");
        auto it1 =
            std::find(std::begin(instructions_), std::end(instructions_), instr1);
        auto it2 =
            std::find(std::begin(instructions_), std::end(instructions_), instr2);
        auto pos1b = std::distance(instructions_.begin(), it1);
        auto pos2b = std::distance(instructions_.begin(), it2);
        if (pos1b <= pos2b) {
          return;
        }
        instructions_.erase(it1);
        instructions_.insert(it2, instr1);
        */
    }
    
    #[inline] pub fn delete_instruction(&mut self, instr: NodeRef<T,U>)  {
        
        todo!();
        /*
            assert(hasInstruction(instr) && "Instruction not in basic block.");
        instructions_.erase(
            std::remove(instructions_.begin(), instructions_.end(), instr),
            instructions_.end());
        untrackNode(instr);
        */
    }
}

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

/// \brief Deletes a referenced node from the control flow graph.
#[inline] pub fn delete_node<G: GraphType>(
    cfg: *mut ControlFlowGraph<G>, 
    node: <G as GraphType>::NodeRef)  {

    todo!();
    /*
        for (auto bbNode : cfg->getMutableNodes()) {
        auto bb = bbNode->data().get();
        if (bb->hasInstruction(node)) {
          bb->deleteInstruction(node);
        }
      }
    */
}
