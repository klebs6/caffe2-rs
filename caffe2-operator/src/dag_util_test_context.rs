crate::ix!();

pub struct DagUtilTestContext {
    net_def:         Arc<NetDef>, // default = nullptr
    operator_nodes:  Vec<OperatorNode>,
}

impl DagUtilTestContext {
    
    pub fn new(spec: &String, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            net_def_ = std::make_shared<NetDef>();
        CAFFE_ENFORCE(TextFormat::ParseFromString(spec, net_def_.get()));
        operator_nodes_ = dag_utils::prepareOperatorNodes(net_def_, ws);
        */
    }
    
    #[inline] pub fn compute_chains(&mut self) -> ExecutionChains {
        
        todo!();
        /*
            return dag_utils::computeGroups(operator_nodes_);
        */
    }
}
