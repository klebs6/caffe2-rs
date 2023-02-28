crate::ix!();

use crate::{
    NetDef,
    OperatorStorage,
    OperatorDef,
    QuantizationFactory,
    TensorQuantizationParams,
};

pub struct QuantizationErrorStats {
    sum_sq:         f32,
    sum_err_sq:     f32,
    max_abs_err:    f32,

    /// actual and reference values 
    /// that resulted in max_abs_err
    max_err_actual: f32,
    max_err_ref:    f32,
    measure_cnt:    i32,
}

#[inline] pub fn has_dnn_lowp_engine_(op_def: &OperatorDef) -> bool {
    
    todo!();
    /*
        const string ENGINE_PREFIX = "DNNLOWP";
      return strncmp(
                 op_def.engine().c_str(),
                 ENGINE_PREFIX.c_str(),
                 ENGINE_PREFIX.size()) == 0;
    */
}


#[inline] pub fn has_dnnlow_pengine_(op: &OperatorStorage) -> bool {
    
    todo!();
    /*
        return HasDNNLowPEngine_(op.debug_def());
    */
}

/**
  | Let consumers of op know that qparams
  | the quantization parameter used for
  | output_index'th output of op.
  |
  */
#[inline] pub fn propagate_output_tensor_quantization_params(
    op:      *mut OperatorStorage,
    idx:     i32,
    qparams: &TensorQuantizationParams)  
{
    
    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
      Int8TensorCPU* output =
          op->Outputs()[idx]->template GetMutable<Int8TensorCPU>();
      output->scale = qparams.scale;
      output->zero_point = qparams.zero_point;
    */
}

/**
  | If input_index'th input is already
  | quantized, return quantization parameter
  | used for the input tensor (should've
  | been set by
  | 
  | PropagateOutputTensorQuantizationParams
  | when the producer was invoked).
  | 
  | If the input tensor is not quantized,
  | return the quantization parameter
  | chosen by qfactory based on the distribution
  | of the input tensor
  |
  */
#[inline] pub fn get_input_tensor_quantization_params_of(
    op:        *mut OperatorStorage,
    idx:       i32,
    qfactory:  *const QuantizationFactory,
    is_weight: Option<bool>) -> TensorQuantizationParams 
{

    let is_weight: bool = is_weight.unwrap_or(false);

    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));

      if (op->InputIsType<Int8TensorCPU>(idx)) {
        const Int8TensorCPU& int8_tensor = op->Input<Int8TensorCPU>(idx);
        TensorQuantizationParams qparams;
        qparams.scale = int8_tensor.scale;
        qparams.zero_point = int8_tensor.zero_point;
        qparams.precision = qfactory->GetActivationPrecision();
        return qparams;
      } else {
        const TensorCPU* tensor = &op->template Input<Tensor>(idx, CPU);
        CAFFE_ENFORCE(tensor->template IsType<float>());
        CAFFE_ENFORCE(tensor->numel() == 0 || tensor->template data<float>());

        float min, max;
        fbgemm::FindMinMax(
            tensor->template data<float>(), &min, &max, tensor->numel());
        auto activation_quantization_kind = qfactory->GetActivationKind();
        if (activation_quantization_kind !=
            QuantizationFactory::QuantizationKind::MIN_MAX_QUANTIZATION) {
          LOG(WARNING)
              << "DNNLOWP dynamic int8 FC uses min_max as the only activation_quantization kind. Qparams will be assigned based on min_max regardless of activation_quantization_kind args.";
        }
        if (is_weight) {
          auto weight_quantization_kind = qfactory->GetWeightKind();
          if (weight_quantization_kind !=
              QuantizationFactory::QuantizationKind::MIN_MAX_QUANTIZATION) {
            LOG(WARNING)
                << "DNNLOWP dynamic int8 FC weight is not constant, assigning qparams to weight based on min_max, regardless of weight_quantization_kind args.";
          }
        }
        return qfactory->ChooseQuantizationParams(min, max, is_weight);
      }
    */
}

#[inline] pub fn output_argument_idx_string_(idx: i32) -> String {
    
    todo!();
    /*
        return idx == 0 ? "" : to_string(idx + 1);
    */
}

#[inline] pub fn output_scale_argument_name(idx: i32) -> String {
    
    todo!();
    /*
        return "Y" + OutputArgumentIdxString_(idx) + "_scale";
    */
}

#[inline] pub fn output_zero_point_argument_name(idx: i32) -> String {
    
    todo!();
    /*
        return "Y" + OutputArgumentIdxString_(idx) + "_zero_point";
    */
}


#[inline] pub fn set_static_quantization_params_(
    op_def:       *mut OperatorDef,
    output_index: i32,
    qparams:      &TensorQuantizationParams)  
{
    
    todo!();
    /*
        AddArgument<float>(
          OutputScaleArgumentName(output_index), qparams.scale, op_def);
      AddArgument<int32_t>(
          OutputZeroPointArgumentName(output_index), qparams.zero_point, op_def);
    */
}


#[inline] pub fn set_static_quantization_params(
    op:           *mut OperatorStorage,
    output_index: i32,
    qparams:      &TensorQuantizationParams)  
{
    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
      auto op_def = make_shared<OperatorDef>();
      *op_def = op->debug_def();
      SetStaticQuantizationParams_(op_def.get(), output_index, qparams);
      op->set_debug_def(op_def);
    */
}

/**
  | -----------
  | @return
  | 
  | true if op's outputs should use static
  | quantization (i.e. op has
  | 
  | Y_scale and optionally Y_zero_offset
  | argument).
  |
  */
#[inline] pub fn has_static_quantization(
    op: *const OperatorStorage,
    output_index: Option<i32>) -> bool 
{
    let output_index: i32 = output_index.unwrap_or(0);

    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
      return op->HasSingleArgumentOfType<float>(
          OutputScaleArgumentName(output_index));
    */
}

/**
  | Get output_index'th quantization
  | parameter.
  | 
  | Should be used only when UseStaticQuantization
  | is true
  |
  */
#[inline] pub fn get_static_quantization_params_of(
    op: *const OperatorStorage,
    idx: i32) -> TensorQuantizationParams 
{
    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));
      unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);

      TensorQuantizationParams qparams;
      qparams.scale = op->GetSingleArgument<float>(OutputScaleArgumentName(idx), 0);
      qparams.zero_point =
          op->GetSingleArgument<int32_t>(OutputZeroPointArgumentName(idx), 0);
      qparams.precision = qfactory->GetActivationPrecision();

      return qparams;
    */
}

/**
  | Quantize input_index'th input if it's
  | not already quantized.
  | 
  | a vector temp should be passed to store
  | quantized results.
  | 
  | -----------
  | @return
  | 
  | array of quantized values for u8, i8,
  | u16, i16
  |
  */
#[inline] pub fn quantize_input_if_needed<T>(
    op:            *mut OperatorStorage,
    input_index:   i32,
    qparams:       &TensorQuantizationParams,
    temp:          &mut Vec<T>) -> *const T 
{
    todo!();
    /*
        if (op->InputIsType<int8::Int8TensorCPU>(input_index)) {
        // Already quantized
        return op->Input<int8::Int8TensorCPU>(input_index).t.data<T>();
      } else {
        // Need to quantize
        const TensorCPU& tensor = op->Input<Tensor>(input_index, CPU);
        temp.resize(tensor.numel());
        fbgemm::Quantize<T>(
            tensor.data<float>(), temp.data(), temp.size(), qparams);
        return temp.data();
      }
    */
}

///for u8, u16
#[inline] pub fn row_wise_quantize_input_if_needed<T>(
    op:           *mut OperatorStorage,
    input_index:  i32,
    qparams:      &Vec<TensorQuantizationParams>,
    temp:         &mut Vec<T>) -> *const T 
{
    todo!();
    /*
        if (op->InputIsType<int8::Int8TensorCPU>(input_index)) {
        // Already quantized
        return op->Input<int8::Int8TensorCPU>(input_index).t.data<T>();
      } else {
        // Need to quantize
        const TensorCPU& tensor = op->Input<Tensor>(input_index, CPU);
        temp.resize(tensor.numel());
        // number of rows
        int N = qparams.size();
        int rowwidth = temp.size() / N;
        // quantize each row
        for (int i = 0; i < N; i++) {
          fbgemm::Quantize<T>(
              tensor.data<float>() + rowwidth * i,
              temp.data() + rowwidth * i,
              rowwidth,
              qparams[i]);
        }
        return temp.data();
      }
    */
}

#[inline] pub fn measure_quantization_error(
    actual: *const f32,
    ref_:   *const f32,
    len:    usize,
    stat:   *mut QuantizationErrorStats)  
{
    todo!();
    /*
        for (int i = 0; i < len; ++i) {
        stat->sum_sq += ref[i] * ref[i];
        float err = actual[i] - ref[i];
        stat->sum_err_sq += err * err;

        if (fabs(err) > stat->max_abs_err) {
          stat->max_abs_err = fabs(err);
          stat->max_err_actual = actual[i];
          stat->max_err_ref = ref[i];
        }
      }
      ++stat->measure_cnt;
    */
}

#[inline] pub fn report_quantization_error(
    op: *const OperatorStorage,
    stat: &QuantizationErrorStats)  
{
    todo!();
    /*
        if (stat.sum_sq == 0) {
        LOG(INFO) << " output " << op->debug_def().output(0) << " of operator "
                  << op << " with type " << op->debug_def().type() << " and engine "
                  << op->debug_def().engine()
                  << " has l2 relative error nan (stat.sum_err_sq "
                  << stat.sum_err_sq << " stat.sum_sq 0)"
                  << " and max abs error " << stat.max_abs_err << " (reference is "
                  << stat.max_err_ref << " and actual is " << stat.max_err_actual
                  << ")"
                  << " sum_err_sq " << stat.sum_err_sq << " sum_sq_ " << stat.sum_sq
                  << " cnt " << stat.measure_cnt;
      } else {
        LOG(INFO) << " output " << op->debug_def().output(0) << " of operator "
                  << op << " with type " << op->debug_def().type() << " and engine "
                  << op->debug_def().engine() << " has l2 relative error "
                  << std::sqrt(stat.sum_err_sq) / std::sqrt(stat.sum_sq)
                  << " and max abs error " << stat.max_abs_err << " (reference is "
                  << stat.max_err_ref << " and actual is " << stat.max_err_actual
                  << ")"
                  << " sum_err_sq " << stat.sum_err_sq << " sum_sq_ " << stat.sum_sq
                  << " cnt " << stat.measure_cnt;
      }
    */
}

#[inline] pub fn get_quantization_factory_of_(
    op_def: &OperatorDef) -> Box<QuantizationFactory> {
    
    todo!();
    /*
        int activation_precision =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(
              op_def,
              "activation_precision",
              FLAGS_caffe2_dnnlowp_activation_quantization_precision);
      int weight_precision = ArgumentHelper::GetSingleArgument<OperatorDef, int>(
          op_def,
          "weight_precision",
          FLAGS_caffe2_dnnlowp_weight_quantization_precision);
      int requantization_multiplier_precision =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(
              op_def,
              "requantization_multiplier_precision",
              FLAGS_caffe2_dnnlowp_requantization_multiplier_precision);
      int eltwise_quantization_precision =
          ArgumentHelper::GetSingleArgument<OperatorDef, int>(
              op_def,
              "eltwise_quantization_precision",
              FLAGS_caffe2_dnnlowp_eltwise_quantization_precision);
      bool preserve_activation_sparsity =
          ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
              op_def,
              "preserve_activation_sparsity",
              FLAGS_caffe2_dnnlowp_preserve_activation_sparsity);
      bool preserve_weight_sparsity =
          ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
              op_def,
              "preserve_weight_sparsity",
              FLAGS_caffe2_dnnlowp_preserve_weight_sparsity);
      bool force_scale_power_of_two =
          ArgumentHelper::GetSingleArgument<OperatorDef, bool>(
              op_def,
              "force_scale_power_of_two",
              FLAGS_caffe2_dnnlowp_force_scale_power_of_two);
      string activation_quantization_kind =
          ArgumentHelper::GetSingleArgument<OperatorDef, string>(
              op_def,
              "activation_quantization_kind",
              FLAGS_caffe2_dnnlowp_activation_quantization_kind);
      string weight_quantization_kind =
          ArgumentHelper::GetSingleArgument<OperatorDef, string>(
              op_def,
              "weight_quantization_kind",
              FLAGS_caffe2_dnnlowp_weight_quantization_kind);
      float weight_p99_threshold =
          ArgumentHelper::GetSingleArgument<OperatorDef, float>(
              op_def,
              "weight_p99_threshold",
              FLAGS_caffe2_dnnlowp_weight_p99_threshold);
      float activation_p99_threshold =
          ArgumentHelper::GetSingleArgument<OperatorDef, float>(
              op_def,
              "activation_p99_threshold",
              FLAGS_caffe2_dnnlowp_activation_p99_threshold);
      std::stringstream ss;
      ss << "Quantization method for op with output " << op_def.output(0)
         << " engine " << op_def.engine() << " activation_precision "
         << activation_precision << " weight_precision " << weight_precision
         << " requantization_multiplier_precision "
         << requantization_multiplier_precision
         << " eltwise_quantization_precision " << eltwise_quantization_precision
         << " preserve_activation_sparsity " << preserve_activation_sparsity
         << " preserve_weight_sparsity " << preserve_weight_sparsity
         << " force_scale_power_of_two " << force_scale_power_of_two
         << " activation_quantization_kind " << activation_quantization_kind
         << " weight_quantization_kind " << weight_quantization_kind;
      if (weight_quantization_kind == "p99" || weight_quantization_kind == "P99") {
        ss << " weight p99 threshold " << weight_p99_threshold;
      }
      if (activation_quantization_kind == "p99" ||
          activation_quantization_kind == "P99") {
        ss << " activation p99 threshold " << activation_p99_threshold;
      }
      VLOG(2) << ss.str();

      return unique_ptr<QuantizationFactory>(new QuantizationFactory(
          activation_precision,
          weight_precision,
          requantization_multiplier_precision,
          eltwise_quantization_precision,
          preserve_activation_sparsity,
          preserve_weight_sparsity,
          force_scale_power_of_two,
          StringToKind(activation_quantization_kind),
          StringToKind(weight_quantization_kind),
          weight_p99_threshold,
          activation_p99_threshold));
    */
}

/**
  | Get QuantizationFactory based on the
  | arguments of op
  |
  */
#[inline] pub fn get_quantization_factory_of(op: *const OperatorStorage) -> Box<QuantizationFactory> {
    
    todo!();
    /*
        return GetQuantizationFactoryOf_(op->debug_def());
    */
}

#[inline] pub fn adjust_output_tensor_quantization_params_with_followed_by(
    op: *mut OperatorStorage,
    followed_by: &String)  
{
    todo!();
    /*
        LOG_IF(WARNING, !HasDNNLowPEngine_(*op));

      auto op_def = make_shared<OperatorDef>();
      *op_def = op->debug_def();
      AddArgument<string>("followed_by", followed_by, op_def.get());
      op->set_debug_def(op_def);

      if (followed_by == "Sigmoid") {
        SetStaticQuantizationParams(
            op, 0, Sigmoid<uint8_t>().GetInputQuantizationParams());
      } else if (followed_by == "Tanh") {
        SetStaticQuantizationParams(
            op, 0, Tanh<uint8_t>().GetInputQuantizationParams());
      } else if (followed_by == "Relu") {
        if (HasStaticQuantization(op)) {
          unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);
          TensorQuantizationParams qparams = GetStaticQuantizationParamsOf(op, 0);
          qparams = qfactory->ChooseQuantizationParams(0, qparams.Max());
          SetStaticQuantizationParams(op, 0, qparams);
        }
      } else {
        LOG(WARNING) << "Unknown followed_by " << followed_by;
      }
    */
}

#[inline] pub fn parse_dnn_lowp_operator_arguments(
    op:                         *mut OperatorStorage,
    dequantize_output:          *mut bool,
    measure_quantization_error: *mut bool,
    followed_by:                *mut String)  
{
    todo!();
    /*
        // When exiting quantized region or we're just doing per-op quantization,
      // dequantize the outputs as floats.
      if (dequantize_output) {
        *dequantize_output =
            op->GetSingleArgument<bool>("dequantize_output", false);
        if (*dequantize_output) {
          VLOG(2) << "Dequantize output " << op->debug_def().output(0)
                  << " of operator type " << op->debug_def().type();
        }
      }

      // Measure quantization error by comparing with reference fp32 operators.
      if (measure_quantization_error) {
        *measure_quantization_error =
            op->GetSingleArgument<bool>("measure_quantization_error", false);
      }

      // Output scale and zero_point can be specified (actually recommended to be
      // specified for performance to avoid on-the-fly quantization parameter
      // selection) from activation distributions collected from profiling.
      if (HasStaticQuantization(op)) {
        TensorQuantizationParams qparams = GetStaticQuantizationParamsOf(op, 0);
        unique_ptr<QuantizationFactory> qfactory = GetQuantizationFactoryOf(op);
        if (qparams.zero_point != (1 << (qfactory->GetActivationPrecision() - 1)) &&
            qparams.zero_point != 0 && qfactory->GetPreserveActivationSparsity()) {
          LOG(WARNING) << "Symmetric quantization is used for activation but "
                          "Y_zero_point is "
                       << qparams.zero_point << " for " << op->debug_def().output(0)
                       << " output activation of an operator with type "
                       << op->debug_def().type();
        }
      } else {
        if (op->HasSingleArgumentOfType<int>("Y_zero_point")) {
          LOG(WARNING) << "Y_zero_point without Y_scale for "
                       << op->debug_def().output(0)
                       << " (an output of operator type " << op->debug_def().type()
                       << ") doesn't make sense";
        }
      }

      // When an operator has only one consumer and the consumer only cares about
      // a limited range of values, we can quantize more precisely.
      if (op->HasSingleArgumentOfType<string>("followed_by")) {
        string followed_by_ = op->GetSingleArgument<string>("followed_by", "");
        VLOG(2) << "Operator with type " << op->debug_def().type() << " and output "
                << op->debug_def().output(0) << " is followed by " << followed_by_;

        AdjustOutputTensorQuantizationParamsWithFollowedBy(op, followed_by_);
        if (followed_by) {
          *followed_by = followed_by_;
        }
      }
    */
}

#[inline] pub fn add_scale_zero_offset_arguments_with_histogram(
    net_def:             NetDef,
    histogram_file_name: &String) -> NetDef 
{
    todo!();
    /*
        ifstream f(histogram_file_name);

      // check the format by looking at the first line
      string first_line, word;
      getline(f, first_line);
      f.seekg(0, f.beg);
      istringstream ist(first_line);
      int nwords_first_line = 0;
      while (ist >> word) {
        ++nwords_first_line;
      }

      ist.str(first_line);
      ist.clear();

      bool new_format = true;
      int op_index, i, nbins;
      string op_type, tensor_name;
      float min, max;
      ist >> op_index >> op_type >> i >> tensor_name >> min >> max >> nbins;
      if (nwords_first_line != nbins + 7) {
        ist.str(first_line);
        ist.clear();
        ist >> op_index >> i >> tensor_name >> min >> max >> nbins;
        if (nwords_first_line == nbins + 6) {
          new_format = false;
        } else {
          LOG(WARNING) << "histogram file " << histogram_file_name
                       << " has an invalid format";
          return net_def;
        }
      }

      // parse the input file
      op_index = 0;
      for (auto& op_def : *net_def.mutable_op()) {
        ArgumentHelper arg_helper(op_def);

        for (i = 0; i < op_def.output().size(); ++i) {
          int op_index2, i2;

          if (new_format) {
            f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max >> nbins;
          } else {
            f >> op_index2 >> i2 >> tensor_name >> min >> max >> nbins;
          }
          LOG_IF(WARNING, op_index2 != op_index)
              << "op index " << op_index2 << " doesn't match with " << op_index;
          LOG_IF(WARNING, tensor_name != op_def.output(i))
              << tensor_name << " in histogram file line " << op_index
              << " doesn't match with operation def " << op_def.output(i);
          LOG_IF(WARNING, i2 != i)
              << "output tensor index " << i2 << " doesn't match with " << i;
          if (new_format) {
            LOG_IF(WARNING, op_type != op_def.type())
                << "operator type " << op_type << " in histogram file line "
                << op_index << " doesn't match with operation def "
                << op_def.type();
          }

          vector<uint64_t> bins;
          for (int j = 0; j < nbins; ++j) {
            uint64_t cnt;
            f >> cnt;
            bins.push_back(cnt);
          }

          if (!HasDNNLowPEngine_(op_def) ||
              arg_helper.GetSingleArgument<int>("dequantize_output", 0) != 0 ||
              i > 0) {
            LOG(INFO) << "Skip " << op_def.type() << " " << op_def.output(0);
            continue;
          }

          Histogram hist = Histogram(min, max, bins);

          unique_ptr<QuantizationFactory> qfactory =
              GetQuantizationFactoryOf_(op_def);
          TensorQuantizationParams qparams =
              qfactory->ChooseQuantizationParams(hist);

          SetStaticQuantizationParams_(&op_def, 0, qparams);
        }
        ++op_index;
      }

      return net_def;
    */
}
