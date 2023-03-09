crate::ix!();

impl<Context> WhileOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 
        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<NetDef>("loop_net"),
            "loop_net must be specified in While operator");
        loop_net_def_ =
            this->template GetSingleArgument<NetDef>("loop_net", NetDef());
        loop_net_ = CreateNet(loop_net_def_, ws);
        CAFFE_ENFORCE(loop_net_, "Failed to initialize loop subnet");

        cond_net_ = nullptr;
        bool has_cond_net =
            this->template HasSingleArgumentOfType<NetDef>("cond_net");
        if (has_cond_net) {
          cond_net_def_ =
              this->template GetSingleArgument<NetDef>("cond_net", NetDef());
          cond_net_ = CreateNet(cond_net_def_, ws);
          CAFFE_ENFORCE(cond_net_, "Failed to initialize condition subnet");
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            this->InputIsTensorType(0, Context::GetDeviceType()),
            "Invalid condition in While operator: tensor expected");

        const auto& condition = Input(0);
        CAFFE_ENFORCE_EQ(
            condition.numel(),
            1,
            "Invalid condition tensor in While operator: single value expected");

        while (true) {
          if (cond_net_ && !cond_net_->Run()) {
            return false;
          }
          if (!*condition.template data<bool>()) {
            return true;
          }
          if (!loop_net_->Run()) {
            return false;
          }
        }

        return true;
        */
    }
}
