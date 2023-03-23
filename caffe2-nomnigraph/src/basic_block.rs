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
