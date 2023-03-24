crate::ix!();

pub trait DebugInfoString {

    #[inline] fn debug_info_string(&self) -> String {
        
        todo!();
        /*
            return "";
        */
    }
}

pub trait GetDebugDef {

    #[inline] fn debug_def(&self) -> &OperatorDef {
        
        todo!();
        /*
            CAFFE_ENFORCE(has_debug_def(), "operator_def was null!");
        return *operator_def_;
        */
    }
}

pub trait SetDebugDef {

    #[inline] fn set_debug_def(&mut self, operator_def: &Arc<OperatorDef>)  {
        
        todo!();
        /*
            operator_def_ = operator_def;
        */
    }
}

pub trait CheckHasDebugDef {
    #[inline] fn has_debug_def(&self) -> bool {
        
        todo!();
        /*
            return operator_def_ != nullptr;
        */
    }
}
