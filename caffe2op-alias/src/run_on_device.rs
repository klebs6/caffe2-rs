crate::ix!();

impl<Context> AliasWithNameOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          name_(this->template GetSingleArgument<std::string>(
                  "name",
                  "invalid_name")),
                  is_backward_(
                      this->template GetSingleArgument<bool>("is_backward", false)) 

                      CAFFE_ENFORCE(
                          OperatorStorage::HasArgument("name"), "You have to specify argument name");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");

        // This doesn't work anymore as this is "newstyle" operator
        // OutputTensorAlias(0, input);

        OperatorStorage::SetOutputTensor(0, input.Alias());
        return true;
        */
    }
}
