crate::ix!();

/**
  | Update the given StatRegistry, or the
  | global StatRegistry, with the values
  | of counters for the given keys.
  |
  */
pub struct StatRegistryUpdateOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{StatRegistryUpdate, StatRegistryUpdateOp}

num_inputs!{StatRegistryUpdate, (2,3)}

num_outputs!{StatRegistryUpdate, 0}

inputs!{
    StatRegistryUpdate, 
    0 => ("keys",   "1D string tensor with the key names to update."),
    1 => ("values", "1D int64 tensor with the values to update."),
    2 => ("handle", "If provided, update the given StatRegistry. Otherwise, update the global singleton.")
}

impl StatRegistryUpdateOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& keys = Input(0);
        const auto& values = Input(1);
        auto registry = InputSize() == 3
            ? OperatorStorage::Input<std::unique_ptr<StatRegistry>>(2).get()
            : &StatRegistry::get();
        CAFFE_ENFORCE_EQ(keys.numel(), values.numel());
        ExportedStatList data(keys.numel());
        auto* pkeys = keys.data<std::string>();
        auto* pvals = values.data<int64_t>();
        int i = 0;
        for (auto& stat : data) {
          stat.key = pkeys[i];
          stat.value = pvals[i];
          ++i;
        }
        registry->update(data);
        return true;
        */
    }
}
