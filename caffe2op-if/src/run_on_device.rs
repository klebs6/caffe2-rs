crate::ix!();

impl<Context> IfOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 

        CAFFE_ENFORCE(
            this->template HasSingleArgumentOfType<NetDef>("then_net"),
            "then_net must be specified in If operator");
        auto then_net_def =
            this->template GetSingleArgument<NetDef>("then_net", NetDef());
        then_net_ = CreateNet(then_net_def, ws);
        CAFFE_ENFORCE(then_net_, "Failed to initialize then subnet");

        if (this->template HasSingleArgumentOfType<NetDef>("else_net")) {
          auto else_net_def =
              this->template GetSingleArgument<NetDef>("else_net", NetDef());
          else_net_ = CreateNet(else_net_def, ws);
          CAFFE_ENFORCE(else_net_, "Failed to initialize else subnet");
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            this->InputIsTensorType(0, Context::GetDeviceType()),
            "Invalid condition in If operator: tensor expected");

        const auto& condition = Input(0);
        CAFFE_ENFORCE_EQ(
            condition.numel(),
            1,
            "Invalid condition tensor in If operator: single value expected");

        auto conditionValue = *condition.template data<bool>();
        if (conditionValue) {
          return then_net_->Run();
        } else if (else_net_) {
          return else_net_->Run();
        }

        return true;
        */
    }
}
