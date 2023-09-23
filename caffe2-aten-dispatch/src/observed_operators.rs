crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/ObservedOperators.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/dispatch/ObservedOperators.cpp]

pub struct ObservedOperators {

}

impl ObservedOperators {
    
    pub fn get_unobserved_operator_list(&mut self) -> &mut HashSet<String> {
        
        todo!();
        /*
            // names of the operators that should not be observed
      static unordered_set<string> not_observed_ops = {
        "size",
        "is_leaf",
        "output_nr",
        "_version",
        "is_complex",
        "profiler::_record_function_enter",
        "profiler::_record_function_exit",
      };
      return not_observed_ops;
        */
    }
    
    pub fn is_observed(&mut self, name: &OperatorName) -> bool {
        
        todo!();
        /*
            return !ObservedOperators::getUnobservedOperatorList().count(name.name);
        */
    }
}
