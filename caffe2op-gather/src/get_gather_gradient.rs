crate::ix!();

pub struct GetGatherGradient;

impl GetGradientDefs for GetGatherGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        const bool dense_gradient =
            argsHelper.GetSingleArgument<bool>("dense_gradient", false);
        const int axis = argsHelper.GetSingleArgument<int>("axis", 0);

        // TBD: While it hasn't been used yet, we need to add wrap_indices support
        // to gradients next.
        // if (argsHelper.HasArgument("wrap_indices_")) {
        // }

        using Op = GatherOp<CPUContext>;

        if (axis == 0) {
          if (dense_gradient) {
            return vector<OperatorDef>{CreateOperatorDef(
                "SparseToDense",
                "",
                vector<string>{I(Op::INDICES), GO(0), I(Op::DATA)},
                vector<string>{GI(Op::DATA)})};
          } else {
            // For now we don't do any reshaping as the consumer of this op would
            // probably be ScatterUpdate which is intenionally ignores shapes. We
            // might need to revisit it in the future for correctness purposes. The
            // right shape for the output woild be to flatten INDICES and collapse
            // first X dims of GRAD
            SetSparse(Op::DATA, I(Op::INDICES), GO(0));
            return vector<OperatorDef>();
          }
        }

        // TBD: This is misleading to use dense_gradient by default for axis 0
        // and not othewise....
        if (argsHelper.HasArgument("dense_gradient")) {
          CAFFE_ENFORCE(
              dense_gradient == true,
              "Gather with axis > 0 must use dense_gradient");
        }

        Argument axisArg = MakeArgument<int>("axis", axis);
        return SingleGradientDef(
            "BatchGatherGradient",
            "",
            // This is the order as expected by BatchGatherGradient indices,
            // different from SpartseToDense above.
            vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
            vector<string>{GI(0)},
            std::vector<Argument>{axisArg});
        */
    }
}

register_gradient!{Gather, GetGatherGradient}
