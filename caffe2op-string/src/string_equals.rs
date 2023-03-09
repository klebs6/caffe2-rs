crate::ix!();

pub struct StrEquals {
    text:  String,
}

impl StrEquals {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : text_(op.GetSingleArgument<std::string>("text", ""))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> bool {
        
        todo!();
        /*
            return str == text_;
        */
    }
}

pub struct Prefix {
    length:  i32,
}

impl Prefix {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : length_(op.GetSingleArgument<int>("length", 3))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> String {
        
        todo!();
        /*
            return std::string(str.begin(), std::min(str.end(), str.begin() + length_));
        */
    }
}

pub struct Suffix {
    length: i32,
}

impl Suffix {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : length_(op.GetSingleArgument<int>("length", 3))
        */
    }
    
    #[inline] pub fn invoke(&mut self, str: &String) -> String {
        
        todo!();
        /*
            return std::string(std::max(str.begin(), str.end() - length_), str.end());
        */
    }
}
