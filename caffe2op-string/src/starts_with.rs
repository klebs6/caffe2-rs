crate::ix!();

pub struct StartsWith {
    prefix:  String,
}

impl StartsWith {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : prefix_(op.GetSingleArgument<std::string>("prefix", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return std::mismatch(prefix_.begin(), prefix_.end(), str.begin()).first ==
            prefix_.end();
        */
    }
}

pub struct EndsWith {
    suffix:  String,
}

impl EndsWith {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : suffix_(op.GetSingleArgument<std::string>("suffix", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return std::mismatch(suffix_.rbegin(), suffix_.rend(), str.rbegin())
                   .first == suffix_.rend();
        */
    }
}
