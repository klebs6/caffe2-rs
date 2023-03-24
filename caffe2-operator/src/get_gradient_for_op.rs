crate::ix!();

declare_registry!{
    GradientRegistry,
    GradientMakerBase,
    OperatorDef,
    Vec<GradientWrapper>
}

/**
  | -----------
  | @brief
  | 
  | Gets the GradientOpsMeta for the given
  | operator def.
  |
  */
#[inline] pub fn get_gradient_for_op(
    def: &OperatorDef,
    g_output: &Vec<GradientWrapper>) -> GradientOpsMeta 
{
    todo!();
    /*
        C10_LOG_API_USAGE_ONCE("caffe2.gradient_maker");
      std::unique_ptr<GradientMakerBase> maker(
          GradientRegistry()->Create(def.type(), def, g_output));
      CAFFE_ENFORCE(
          maker, "Gradient maker for operator ", def.type(), " not implemented.");
      GradientOpsMeta meta = maker->Get();
      // Copy device option, engine, and arguments if needed.
      if (maker->CopyDeviceOption() && def.has_device_option()) {
        for (OperatorDef& grad_def : meta.ops_) {
          grad_def.mutable_device_option()->CopyFrom(def.device_option());
        }
      }
      // Copy engine if needed.
      if (maker->CopyEngine() && def.has_engine()) {
        for (OperatorDef& grad_def : meta.ops_) {
          grad_def.set_engine(def.engine());
        }
      }
      // Copy arguments if needed.
      if (maker->CopyArguments() && def.arg_size()) {
        for (OperatorDef& grad_def : meta.ops_) {
          for (auto& arg : def.arg()) {
            grad_def.add_arg()->CopyFrom(arg);
          }
        }
      }
      // VLOG for debugging purposes.
      for (const OperatorDef& grad_def : meta.ops_) {
        VLOG(1) << "Gradient ops: " << ProtoDebugString(grad_def);
      }
      // Check if the gradient computation has returned the right size for the
      // gradient vector.
      CAFFE_ENFORCE_EQ(meta.g_input_.size(), def.input_size());
      VLOG(1) << "Gradients:";
      for (const GradientWrapper& grad : meta.g_input_) {
        // The gradient should either be (1) not set, or (2) dense, or (3) sparse,
        // but cannot be both dense and sparse.
        if (!grad.IsDense() && !grad.IsSparse()) {
          VLOG(1) << "\t [no gradient]";
        } else if (grad.IsDense()) {
          VLOG(1) << "\t [dense]" << grad.dense_;
        } else {
          CAFFE_ENFORCE(
              grad.indices_.size() && grad.values_.size(),
              "For sparse gradient, one should set both indices and values. "
              "Currently we have: (" +
                  grad.indices_ + ", " + grad.values_ + ").");
          VLOG(1) << "\t [sparse] " << grad.indices_ << ", " << grad.values_;
        }
      }
      return meta;
    */
}


