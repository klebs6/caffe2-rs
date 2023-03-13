crate::ix!();


/**
  | Count the number recent update on rows.
  | 
  | Exponential decay is applied on the
  | counter with decay rate r, such that
  | r^{counter_halflife} = 0.5;
  | 
  | If counter_halflife is nonpositive,
  | this operator is turned off.
  |
  */
pub struct RowWiseCounterOp {
    storage: OperatorStorage,
    context: CPUContext,

    counter_halflife:    i64,
    counter_neg_log_rho: f64,
}

num_inputs!{RowWiseCounter, 4}

num_outputs!{RowWiseCounter, 2}

inputs!{RowWiseCounter, 
    0 => ("prev_iter",             "Iter at last update"),
    1 => ("update_counter",        "update counter"),
    2 => ("indices",               "Sparse indices"),
    3 => ("iter",                  "current iteration")
}

outputs!{RowWiseCounter, 
    0 => ("output_prev_iter",      "Updated iter at last update"),
    1 => ("output_update_counter", "Output update counter")
}

args!{RowWiseCounter, 
    0 => ("counter_halflife",      "Default -1: off")
}

enforce_one_to_one_inplace!{RowWiseCounter}

register_cpu_operator!{RowWiseCounter, RowWiseCounterOp}

should_not_do_gradient!{RowWiseCounter}

input_tags!{
    RowWiseCounter
    {
        PrevIter,
        Counter,
        Indices,
        Iter
    }
}

output_tags!{
    RowWiseCounter
    {
        OutputPrevIter,
        OutputCounter
    }
}

impl RowWiseCounterOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            counter_halflife_( this->template GetSingleArgument<int64_t>("counter_halflife", -1)),
            counter_neg_log_rho_(0.0) 

        if (counter_halflife_ > 0) {
          counter_neg_log_rho_ = std::log(2.0) / counter_halflife_;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(Input(PREV_ITER).numel(), Input(COUNTER).numel());
        CAFFE_ENFORCE_EQ(Input(ITER).numel(), 1);
        return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, Input(INDICES));
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self, ) -> bool {
        todo!();
        /*
            auto* prev_iter =
            Output(OUTPUT_PREV_ITER)->template mutable_data<int64_t>();
        auto* counter = Output(OUTPUT_COUNTER)->template mutable_data<double>();

        const int64_t curr_iter = Input(ITER).template data<int64_t>()[0];
        const auto* indices = Input(INDICES).template data<SIndex>();

        auto n = Input(INDICES).numel();
        if (n == 0) {
          return true;
        }
        if (counter_halflife_ <= 0) {
          return true;
        }

        for (auto i = 0; i < n; ++i) {
          const std::size_t idx = indices[i];
          CAFFE_ENFORCE_GE(
              Input(COUNTER).numel(),
              idx,
              this->debug_def().input(COUNTER),
              ", out of bound,  idx:",
              idx,
              " for input i:",
              i,
              " max size:",
              Input(COUNTER).numel());
          const int64_t iter_delta =
              std::max<int64_t>(0, curr_iter - prev_iter[idx]);

          counter[idx] =
              1.0 + std::exp(-iter_delta * counter_neg_log_rho_) * counter[idx];
          prev_iter[idx] = std::max<int64_t>(curr_iter, prev_iter[idx]);
        }
        return true;
        */
    }
}
