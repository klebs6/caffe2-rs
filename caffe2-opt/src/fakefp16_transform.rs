crate::ix!();

define_bool!{
    fake_fp16_conversion_use_fp16_acc,
    false,
    "Whether to enable fp16 accumulation for FC / BatchMatMul for fakefp16 operators."
}

define_bool!{
    fake_fp16_conversion_use_nnpi,
    false,
    "Whether to simulate NNPI behavior for fakefp16 operators."
}

/// Mapping from fp32 ops to fakefp16 ops
#[inline] pub fn get_fake_fp_16op_mapping(
    use_fp16_acc: Option<bool>, 
    use_nnpi:     Option<bool>) -> HashMap<String,String> 
{
    let use_fp16_acc = use_fp16_acc.unwrap_or(false);
    let use_nnpi     = use_nnpi.unwrap_or(false);

    todo!();
    /*
        std::unordered_map<std::string, std::string> fake_fp16_op_conversion_map = {
          {"FC", "Fp16FCAcc32NNPI"},
          {"Int8FC", "Int8FCFakeAcc32NNPI"},
          {"Int8Quantize", "Int8QuantizeNNPI"},
          {"Int8Dequantize", "Int8DequantizeNNPI"},
          {"LayerNorm", "LayerNormFakeFP16NNPI"},
          {"FbFCPacked", "Fp16FCAcc32NNPI"},
          {"Logit", "LogitFakeFp16NNPI"},
          {"SparseLengthsSum", "SparseLengthsSumFakeFP16AccFP16"},
          {"SparseLengthsWeightedSum", "SparseLengthsWeightedSumFakeFP16AccFP16"},
          {"SparseLengthsMean", "SparseLengthsMeanFakeFP16AccFP16"},
          {"SparseLengthsSumFused4BitRowwise",
           "SparseLengthsSumFused4BitRowwiseFakeFP16NNPI"},
          {"SparseLengthsWeightedSumFused4BitRowwise",
           "SparseLengthsWeightedSumFused4BitRowwiseFakeFP16NNPI"},
          {"SparseLengthsSumFused8BitRowwise",
           "SparseLengthsSumFused8BitRowwiseFakeFP16NNPI"},
          {"SparseLengthsWeightedSumFused8BitRowwise",
           "SparseLengthsWeightedSumFused8BitRowwiseFakeFP16NNPI"},
          {"SparseLengthsMeanFused8BitRowwise",
           "SparseLengthsMeanFused8BitRowwiseFakeFP16AccFP16"},
          {"BatchMatMul", "BatchMatMulFP16Acc32Fake"},
          {"Sigmoid", "SigmoidFakeFp16"},
          {"SpatialBN", "SpatialBNFakeFp16NNPI"},
          {"Swish", "SwishFakeFp16NNPI"},
          {"Tanh", "TanhFakeFp16"},
          {"Relu", "ReluFakeFp16"},
          {"Add", "AddFakeFp16"},
          {"Sub", "SubFakeFp16"},
          {"Mul", "MulFakeFp16"},
          {"Div", "DivFakeFp16"},
          {"Sum", "SumFakeFp16"},
          {"Sqr", "SqrFakeFp16"},
          {"LengthsSum", "LengthsSumFakeFp16"}};
      if (use_fp16_acc) {
        fake_fp16_op_conversion_map["FC"] = "Fp16FCAcc16NNPI";
        fake_fp16_op_conversion_map["FbFCPacked"] = "Fp16FCAcc16NNPI";
        fake_fp16_op_conversion_map["BatchMatMul"] = "BatchMatMulFP16Acc16Fake";
      }
      if (use_nnpi) {
        fake_fp16_op_conversion_map["Sigmoid"] = "SigmoidFakeFp16NNPI";
        fake_fp16_op_conversion_map["Tanh"] = "TanhFakeFp16NNPI";
      }
      return fake_fp16_op_conversion_map;
    */
}

#[inline] pub fn find_mutable_operator_by_input(net: *mut NetDef, input: &String) -> Vec<*mut OperatorDef> {
    
    todo!();
    /*
        std::vector<OperatorDef*> ops;

      for (auto& op : *net->mutable_op()) {
        for (const auto& i : op.input()) {
          if (input == i) {
            ops.push_back(&op);
          }
        }
      }
      return ops;
    */
}

#[inline] pub fn fake_fp_16fold_layer_norm(net: *mut NetDef)  {
    
    todo!();
    /*
        for (auto& op : *net->mutable_op()) {
        if (op.type() == "LayerNormFakeFP16NNPI") {
          LOG(INFO) << "Attemping to fuse LayerNormFakeFP16NNPI at "
                    << ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                           op, "net_pos", -1);
          if (op.input().size() != 1) {
            LOG(INFO) << "input isn't 1, skipping";
            continue;
          }

          const std::string& ln_output = op.output(0);
          auto next_ops = findMutableOperatorByInput(net, ln_output);

          if (next_ops.size() != 1 || next_ops[0]->type() != "MulFakeFp16") {
            LOG(INFO) << "next op isn't MulFakeFp16, skipping";
            continue;
          }

          auto* mul_op = next_ops[0];

          auto next_next_ops = findMutableOperatorByInput(net, mul_op->output(0));

          if (next_next_ops.size() != 1 ||
              next_next_ops[0]->type() != "AddFakeFp16") {
            LOG(INFO) << "next op isn't AddFakeFp16, skipping";
            continue;
          }

          auto* add_op = next_next_ops[0];

          *(op.mutable_input()->Add()) = mul_op->input(1);
          *(op.mutable_input()->Add()) = add_op->input(1);
          *op.mutable_output(0) = add_op->output(0);

          mul_op->set_type("delete_me_optimized_away");
          add_op->set_type("delete_me_optimized_away");

          LOG(INFO) << "Fused LayerNormFakeFP16NNPI";
        }
      }
    */
}

#[inline] pub fn fake_fp_16fold_layer_norm_quant(net: *mut NetDef)  {
    
    todo!();
    /*
        for (auto& op : *net->mutable_op()) {
        if (op.type() == "LayerNormFakeFP16NNPI") {
          auto layernormNetPos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                                 op, "net_pos", -1);
          LOG(INFO) << "Attemping to fuse LayerNormFakeFP16NNPI w Quant at "
                    << layernormNetPos;
          if (op.input().size() != 1) {
            LOG(INFO) << "input isn't 1, is " << op.input().size() << " skipping";
            continue;
          }

          const std::string& ln_output = op.output(0);
          auto next_ops = findMutableOperatorByInput(net, ln_output);

          if (next_ops.size() != 1 || next_ops[0]->type() != "Int8QuantizeNNPI") {
            LOG(INFO) << "next op isn't Int8QuantizeNNPI, skipping";
            continue;
          }

          auto* quantOp = next_ops[0];

          if (quantOp->output().size() != 1) {
            LOG(INFO) << "more than one output for quant, skipping";
            continue;
          }

          op.set_type("LayerNormInt8QuantizeFakeNNPI");

          *op.mutable_output(0) = quantOp->output(0);
          op.add_arg()->CopyFrom(MakeArgument("Y_scale",
                          ArgumentHelper::GetSingleArgument<OperatorDef, float>(*quantOp, "Y_scale", -1)));
          op.add_arg()->CopyFrom(MakeArgument("Y_zero_point",
                          ArgumentHelper::GetSingleArgument<OperatorDef, int>(*quantOp, "Y_zero_point", -1)));

          auto quantNetPos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              *quantOp, "net_pos", -1);

          quantOp->set_type("delete_me_optimized_away");

          LOG(INFO) << "Fused LayerNormFakeFP16NNPI w Quant at " << layernormNetPos << " " << quantNetPos;
        }
      }
    */
}

#[inline] pub fn fake_fp_16fold_swish(net: *mut NetDef)  {
    
    todo!();
    /*
        // find a sequence deq->swish->quant and replace it
      for (auto& op : *net->mutable_op()) {
        if (op.type() == "Int8DequantizeNNPI") {
          auto deq_net_pos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              op, "net_pos", -1);

          LOG(INFO) << "Attempting swish fusion at " << deq_net_pos;

          if (op.output().size() != 1) {
            LOG(INFO) << "more than one output deq, skipping";
            continue;
          }

          const std::string& deqOutput = op.output(0);
          auto next_ops = findMutableOperatorByInput(net, deqOutput);

          if (next_ops.size() != 1 || next_ops[0]->type() != "SwishFakeFp16NNPI") {
            LOG(INFO) << "skipping, next op is " << next_ops[0]->type();
            continue;
          }

          auto* swishOp = next_ops[0];

          if (swishOp->output().size() != 1) {
            LOG(INFO) << "more than one output for swish, skipping";
            continue;
          }

          auto next_next_ops = findMutableOperatorByInput(net, swishOp->output(0));

          if (next_next_ops.size() != 1 || next_next_ops[0]->type() != "Int8QuantizeNNPI") {
            LOG(INFO) << "skipping, next op isn't quant, is " << next_next_ops[0]->type();
            continue;
          }

          auto* quantOp = next_next_ops[0];

          op.set_type("SwishFakeInt8NNPI");
          *op.mutable_output(0) = quantOp->output(0);
          op.add_arg()->CopyFrom(MakeArgument("Y_scale",
                          ArgumentHelper::GetSingleArgument<OperatorDef, float>(*quantOp, "Y_scale", -1)));
          op.add_arg()->CopyFrom(MakeArgument("Y_zero_point",
                          ArgumentHelper::GetSingleArgument<OperatorDef, int>(*quantOp, "Y_zero_point", -1)));

          auto swish_net_pos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              *swishOp, "net_pos", -1);
          auto quant_net_pos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              *quantOp, "net_pos", -1);

          swishOp->set_type("delete_me_optimized_away");
          quantOp->set_type("delete_me_optimized_away");

          LOG(INFO) << "Fusing swish at " << deq_net_pos << ", " << swish_net_pos << ", " << quant_net_pos;
        }
      }
    */
}

#[inline] pub fn fake_fp_16fold_tanh_quant(net: *mut NetDef)  {
    
    todo!();
    /*
        // find a sequence deq->swish->quant and replace it
      for (auto& op : *net->mutable_op()) {
        if (op.type() == "TanhFakeFp16NNPI") {
          auto tanh_net_pos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              op, "net_pos", -1);

          LOG(INFO) << "Attempting tanh fusion at " << tanh_net_pos;

          if (op.output().size() != 1) {
            LOG(INFO) << "more than one output for tanh, skipping";
            continue;
          }

          const std::string& tanhOutput = op.output(0);
          auto next_ops = findMutableOperatorByInput(net, tanhOutput);

          if (next_ops.size() != 1 || next_ops[0]->type() != "Int8QuantizeNNPI") {
            LOG(INFO) << "skipping, next op is " << next_ops[0]->type();
            continue;
          }

          auto* quantOp = next_ops[0];

          if (quantOp->output().size() != 1) {
            LOG(INFO) << "more than one output for quant, skipping";
            continue;
          }

          op.set_type("TanhQuantFakeFp16NNPI");
          *op.mutable_output(0) = quantOp->output(0);
          op.add_arg()->CopyFrom(MakeArgument("Y_scale",
                          ArgumentHelper::GetSingleArgument<OperatorDef, float>(*quantOp, "Y_scale", -1)));
          op.add_arg()->CopyFrom(MakeArgument("Y_zero_point",
                          ArgumentHelper::GetSingleArgument<OperatorDef, int>(*quantOp, "Y_zero_point", -1)));

          auto quant_net_pos = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
                              *quantOp, "net_pos", -1);


          quantOp->set_type("delete_me_optimized_away");

          LOG(INFO) << "Fusing tanh and quant at " << tanh_net_pos << ", " << quant_net_pos;
        }
      }
    */
}

#[inline] pub fn fake_fp_16fuse_ops(net: *mut NetDef)  {
    
    todo!();
    /*
        LOG(INFO) << "Running Fp16 Fusion";

      // We should fuse the groups of bigger operators first
      fakeFp16FoldLayerNorm(net);
      fakeFp16FoldSwish(net);
      fakeFp16FoldTanhQuant(net);
      fakeFp16FoldLayerNormQuant(net);

      auto iter = net->mutable_op()->begin();
      while (iter != net->mutable_op()->end()) {
        if (iter->type() == "delete_me_optimized_away") {
          iter = net->mutable_op()->erase(iter);
        } else {
          ++iter;
        }
      }
    */
}

/**
  | Transform normal fp32 operators to
  | fakefp16 operators.
  |
  */
#[inline] pub fn fake_fp_16transform(net: *mut NetDef)  {
    
    todo!();
    /*
        static const std::unordered_map<std::string, std::string>
          kFakeFp16OpConversionMap = getFakeFp16OpMapping(
              FLAGS_fake_fp16_conversion_use_fp16_acc,
              FLAGS_fake_fp16_conversion_use_nnpi);

      auto blocklist_pos = glow::ParseNetPositionList(FLAGS_onnxifi_blacklist);
      auto blocklist_type = glow::ParseBlockListOps(FLAGS_onnxifi_blacklist_ops);

      // A hack to only do fakefp16 transformation for operators which will be
      // lowered to ONNXIFI.
      // TODO(yingz): Use more deterministic logics to figure out operators which
      // can be lowered to ONNXIFI instead.
      int last_clip_idx = -1;
      for (int i = 0; i < net->op().size(); ++i) {
        const auto& op = net->op(i);
        if (op.type() == "Clip") {
          last_clip_idx = i;
        }
      }
      for (int i = 0; i < net->op().size(); ++i) {
        if (i <= last_clip_idx) {
          continue;
        }
        auto* op = net->mutable_op(i);
        auto net_pos =
            ArgumentHelper::GetSingleArgument<OperatorDef, int>(*op, "net_pos", -1);
        if (blocklist_pos.count(net_pos) || blocklist_type.count(op->type())) {
          continue;
        }
        auto it = kFakeFp16OpConversionMap.find(op->type());
        if (it != kFakeFp16OpConversionMap.end()) {
          op->set_type(it->second);
        }
      }

      fakeFp16FuseOps(net);
    */
}

