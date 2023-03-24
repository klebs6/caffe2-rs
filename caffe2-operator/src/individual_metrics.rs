crate::ix!();

pub struct IndividualMetrics<'a> {
    setup_time:                        f32, // default = 0.0
    memory_alloc_time:                 f32, // default = 0.0
    memory_dealloc_time:               f32, // default = 0.0
    output_dealloc_time:               f32, // default = 0.0
    main_runs:                         i32,
    operators:                         &'a Vec<*mut OperatorStorage>,
    time_per_op:                       Vec<f32>,
    flops_per_op:                      Vec<u64>,
    memory_bytes_read_per_op:          Vec<u64>,
    memory_bytes_written_per_op:       Vec<u64>,
    param_bytes_per_op:                Vec<u64>,
    num_ops_per_op_type:               HashMap<String,i32>,
    time_per_op_type:                  HashMap<String,f32>,
    flops_per_op_type:                 HashMap<String,f32>,
    memory_bytes_read_per_op_type:     HashMap<String,f32>,
    memory_bytes_written_per_op_type:  HashMap<String,f32>,
    param_bytes_per_op_type:           HashMap<String,f32>,
}

impl<'a> IndividualMetrics<'a> {

    pub fn new(operators: &Vec<*mut OperatorStorage>) -> Self {
    
        todo!();
        /*
            : main_runs_(0), operators_(operators) 

        const auto num_ops = operators_.size();
        time_per_op.resize(num_ops, 0.0);
        */
    }

    #[inline] pub fn get_time_per_op(&mut self) -> &Vec<f32> {
        
        todo!();
        /*
            return time_per_op;
        */
    }
    
    /// run ops while collecting profiling results
    #[inline] pub fn run_ops_with_profiling(&mut self)  {
        
        todo!();
        /*
            int idx = 0;
      Timer timer;
      for (auto* op : operators_) {
        const string& op_type = op->debug_def().type();
        if (main_runs_ == 0) { // Gather flops on the first run.
          auto* schema = OpSchemaRegistry::Schema(op_type);
          if (schema && schema->HasCostInferenceFunction()) {
            vector<TensorShape> shapes = op->InputTensorShapes();

            auto all_good_shapes = std::accumulate(
                shapes.begin(),
                shapes.end(),
                true,
                [](bool acc, const TensorShape& shape) {
                  return acc && !shape.unknown_shape();
                });
            OpSchema::Cost cost;
            if (all_good_shapes) {
              cost = schema->InferCost(op->debug_def(), shapes);
            }

            flops_per_op.emplace_back(cost.flops);
            memory_bytes_read_per_op.emplace_back(cost.bytes_read);
            memory_bytes_written_per_op.emplace_back(cost.bytes_written);
            param_bytes_per_op.emplace_back(cost.params_bytes);

            flops_per_op_type[op_type] += cost.flops;
            memory_bytes_read_per_op_type[op_type] += cost.bytes_read;
            memory_bytes_written_per_op_type[op_type] += cost.bytes_written;
            param_bytes_per_op_type[op_type] += cost.params_bytes;
          } else {
            flops_per_op.emplace_back(0);
            memory_bytes_read_per_op.emplace_back(0);
            memory_bytes_written_per_op.emplace_back(0);
            param_bytes_per_op.emplace_back(0);
          }
        }
        timer.Start();
        CAFFE_ENFORCE(
            op->Run(),
            "operator ",
            op->debug_def().name(),
            "(",
            op_type,
            ") has failed.");
        float spent = timer.MilliSeconds();
        time_per_op[idx] += spent;
        time_per_op_type[op_type] += spent;
        ++idx;
      }
      ++main_runs_;
        */
    }
    
    /// print out profiling results
    #[inline] pub fn print_operator_profiling_results(&mut self)  {
        
        todo!();
        /*
            for (auto& op : operators_) {
        op->ResetEvent();
      }
      size_t idx = 0;
      for (auto& op : operators_) {
        const string& op_type = op->debug_def().type();
        num_ops_per_op_type_[op_type]++;
        const string& print_name =
            (op->debug_def().name().size()
                 ? op->debug_def().name()
                 : (op->debug_def().output_size() ? op->debug_def().output(0)
                                                  : "NO_OUTPUT"));
        std::stringstream flops_str;
        if (idx < flops_per_op.size() && flops_per_op[idx]) {
          flops_str << " (" << to_string(1.0e-9 * flops_per_op[idx]) << " GFLOP, "
                    << to_string(
                           1.0e-6 * flops_per_op[idx] / time_per_op[idx] *
                           main_runs_)
                    << " GFLOPS)";
        }
        std::stringstream memory_bytes_read_str;
        if (idx < memory_bytes_read_per_op.size() &&
            memory_bytes_read_per_op[idx]) {
          memory_bytes_read_str << " ("
                                << to_string(1.0e-6 * memory_bytes_read_per_op[idx])
                                << " MB)";
        }
        std::stringstream memory_bytes_written_str;
        if (idx < memory_bytes_written_per_op.size() &&
            memory_bytes_written_per_op[idx]) {
          memory_bytes_written_str
              << " (" << to_string(1.0e-6 * memory_bytes_written_per_op[idx])
              << " MB)";
        }
        std::stringstream param_bytes_str;
        if (idx < param_bytes_per_op.size() && param_bytes_per_op[idx]) {
          param_bytes_str << " (" << to_string(1.0e-6 * param_bytes_per_op[idx])
                          << " MB)";
        }
        std::cout << "Operator #" << idx << " (" << print_name << ", " << op_type
                  << ") " << time_per_op[idx] / main_runs_ << " ms/iter"
                  << flops_str.str() << memory_bytes_written_str.str()
                  << param_bytes_str.str() << std::endl;
        ++idx;
      }
      const std::vector<string> metric(
          {"Time",
           "FLOP",
           "Feature Memory Read",
           "Feature Memory Written",
           "Parameter Memory"});
      const std::vector<double> normalizer(
          {1.0 / main_runs_, 1.0e-9, 1.0e-6, 1.0e-6, 1.0e-6});
      const std::vector<string> unit({"ms", "GFLOP", "MB", "MB", "MB"});

      std::vector<CaffeMap<string, float>*> metric_per_op_type_vec_vec;
      metric_per_op_type_vec_vec.emplace_back(&time_per_op_type);
      metric_per_op_type_vec_vec.emplace_back(&flops_per_op_type);
      metric_per_op_type_vec_vec.emplace_back(&memory_bytes_read_per_op_type);
      metric_per_op_type_vec_vec.emplace_back(&memory_bytes_written_per_op_type);
      metric_per_op_type_vec_vec.emplace_back(&param_bytes_per_op_type);
      for (size_t i = 0; i < metric_per_op_type_vec_vec.size(); ++i) {
        auto* item = metric_per_op_type_vec_vec[i];
        std::vector<std::pair<string, float>> metric_per_op_type_vec(
            (*item).begin(), (*item).end());
        std::sort(
            metric_per_op_type_vec.begin(),
            metric_per_op_type_vec.end(),
            PairLargerThan<string, float>);
        float total_metric = 0.;
        for (const auto& op_item : metric_per_op_type_vec) {
          total_metric += op_item.second * normalizer[i];
        }
        if (total_metric > 0.) {
          std::cout << metric[i] << " per operator type:" << std::endl;
        }
        for (const auto& op_item : metric_per_op_type_vec) {
          float percent = 0.;
          const string& op = op_item.first;
          float value = op_item.second;
          if (total_metric > 0.) {
            percent = (100.0 * value * normalizer[i] / total_metric);
          }
          std::cout << std::setw(15) << std::setfill(' ') << value * normalizer[i]
                    << " " << unit[i] << ". " << std::setw(10) << std::setfill(' ')
                    << percent << "%. " << op << " (" << num_ops_per_op_type_[op]
                    << " ops)" << std::endl;
        }
        if (total_metric > 0.) {
          std::cout << std::setw(15) << std::setfill(' ') << total_metric << " "
                    << unit[i] << " in Total" << std::endl;
        }
        if (i == 0) {
          if (setup_time > 0) {
            std::cout << "BlackBoxPredictor setup time: "
                      << setup_time * normalizer[i] << " " << unit[i] << "\n";
          }
          if (memory_alloc_time > 0) {
            std::cout << "Memory allocation time: "
                      << memory_alloc_time * normalizer[i] << " " << unit[i]
                      << "\n";
          }
          if (memory_dealloc_time > 0) {
            std::cout << "Memory deallocation time: "
                      << memory_dealloc_time * normalizer[i] << " " << unit[i]
                      << std::endl;
          }
          if (output_dealloc_time > 0) {
            std::cout << "Output deallocation time: "
                      << output_dealloc_time * normalizer[i] << " " << unit[i]
                      << std::endl;
          }
        }
      }
        */
    }
}

