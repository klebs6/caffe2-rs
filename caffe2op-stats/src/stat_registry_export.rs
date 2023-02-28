crate::ix!();

pub struct StatRegistryExportOp {
    storage: OperatorStorage,
    context: CPUContext,
    reset:   bool,
}

register_cpu_operator!{StatRegistryExport, StatRegistryExportOp}

num_inputs!{StatRegistryExport, (0,1)}

num_outputs!{StatRegistryExport, 3}

inputs!{StatRegistryExport, 
    0 => ("handle", "If provided, export values from given StatRegistry. Otherwise, export values from the global singleton StatRegistry.")
}

outputs!{StatRegistryExport, 
    0 => ("keys", "1D string tensor with exported key names"),
    1 => ("values", "1D int64 tensor with exported values"),
    2 => ("timestamps", "The unix timestamp at counter retrieval.")
}

args!{StatRegistryExport, 
    0 => ("reset", "(default true) Whether to atomically reset the counters afterwards.")
}

impl StatRegistryExportOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            reset_(GetSingleArgument<bool>("reset", true))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto registry = InputSize() > 0
            ? OperatorStorage::Input<std::unique_ptr<StatRegistry>>(0).get()
            : &StatRegistry::get();
        auto* keys = Output(0);
        auto* values = Output(1);
        auto* timestamps = Output(2);
        auto data = registry->publish(reset_);
        keys->Resize(data.size());
        values->Resize(data.size());
        timestamps->Resize(data.size());
        auto* pkeys = keys->template mutable_data<std::string>();
        auto* pvals = values->template mutable_data<int64_t>();
        auto* ptimestamps = timestamps->template mutable_data<int64_t>();
        int i = 0;
        for (const auto& stat : data) {
          pkeys[i] = std::move(stat.key);
          pvals[i] = stat.value;
          ptimestamps[i] =
              std::chrono::nanoseconds(stat.ts.time_since_epoch()).count();
          ++i;
        }
        return true;
        */
    }
}
