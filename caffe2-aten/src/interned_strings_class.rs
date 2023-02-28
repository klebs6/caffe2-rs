crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/interned_strings_class.h]

pub struct InternedStrings {
    string_to_sym: HashMap<String,Symbol>,
    sym_to_info:   Vec<SymbolInfo>,
    mutex:         Mutex,
}

impl InternedStrings {
    
    pub fn symbol(&mut self, s: &String) -> Symbol {
        
        todo!();
        /*
        
        */
    }
    
    pub fn string(&mut self, sym: Symbol) -> Pair<*const u8,*const u8> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn ns(&mut self, sym: Symbol) -> Symbol {
        
        todo!();
        /*
        
        */
    }
 
    /// prereq - holding mutex_
    ///
    pub fn symbol(&mut self, s: &String) -> Symbol {
        
        todo!();
        /*
        
        */
    }
    
    pub fn custom_string(&mut self, sym: Symbol) -> Pair<*const u8,*const u8> {
        
        todo!();
        /*
        
        */
    }
}

pub struct SymbolInfo {
    ns:          Symbol,
    qual_name:   String,
    unqual_name: String,
}
