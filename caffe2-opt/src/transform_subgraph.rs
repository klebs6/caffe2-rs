crate::ix!();

/**
  | @note
  | 
  | subgraph always starts with ops and
  | ends with tensors, except for the very
  | first group, which can be all tensors
  |
  */
pub struct TransformSubgraph<T,U> {

    input_nodes:          Vec<NodeRef<T,U>>,
    nodes:                Vec<NodeRef<T,U>>,
    external_input_refs:  HashMap<String,NodeRef<T,U>>,
    external_output_refs: HashMap<String,NodeRef<T,U>>,
    group_id:             i32, //{-1};
    needed:               bool, //{true};
}

impl<T,U> TransformSubgraph<T,U> {
    
    pub fn new(
        f:    Vec<NodeRef<T,U>>,
        n:    Vec<NodeRef<T,U>>,
        id:   i32,
        need: bool) -> Self 
    {
        todo!();
        /*
            : input_nodes(std::move(f)),
            nodes(std::move(n)),
            group_id(id),
            needed(need)
        */
    }
    
    pub fn new_from_other(rhs: TransformSubgraph<T,U>) -> Self {
        todo!();
        /*
            : input_nodes(std::move(rhs.input_nodes)),
            nodes(std::move(rhs.nodes)),
            external_input_refs(std::move(rhs.external_input_refs)),
            external_output_refs(std::move(rhs.external_output_refs)),
            group_id(rhs.group_id),
            needed(rhs.needed)
        */
    }
    
    #[inline] pub fn print(&self)  {
        
        todo!();
        /*
            LOG(INFO) << "Group :" << group_id;
        LOG(INFO) << "  Input Nodes: ";
        for (const auto i : input_nodes) {
          LOG(INFO) << "    " << ShowNode(i);
        }
        LOG(INFO) << "  Nodes: ";
        for (const auto i : nodes) {
          LOG(INFO) << "    " << ShowNode(i);
        }
        */
    }
}
