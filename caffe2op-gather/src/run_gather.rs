crate::ix!();

impl<Context> GatherOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 0),
            OP_SINGLE_ARG(bool, "match_outer", match_outer_, false) 

        // TBD: We may want to fix the old index wrap behaviour once we have
        // operator versioning, to only apply it when needed as otherwise its likely
        // an error.
        // Right now, we apply index wrapping by default only to axis == 0,
        // since we have ONNX conversion code that uses it. For other ops it
        // needs to be specified explicitly with argument or you don't get it.
        if (OperatorStorage::HasArgument("wrap_indices")) {
          wrap_indices_ = Operator<Context>::template GetSingleArgument<bool>(
              "wrap_indices", (false));
        } else {
          wrap_indices_ = (axis_ == 0) ? true : false;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }

    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            return gather_helper::gather_impl<Index, Context>(
                this, DATA, INDICES, 0, axis_, wrap_indices_, match_outer_);
        */
    }
}
