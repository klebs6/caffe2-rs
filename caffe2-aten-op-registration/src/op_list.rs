crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/ATenOpList.cpp]

pub struct OpNameEquals {

}

impl OpNameEquals {
    
    pub fn invoke(
        &self, 
        lhs: &(*const u8,*const u8),
        rhs: &(*const u8,*const u8)

    ) -> bool {
        
        todo!();
        /*
            return 0 == strcmp(lhs.first, rhs.first) && 0 == strcmp(lhs.second, rhs.second);
        */
    }
}

pub struct OpNameHash {

}

impl OpNameHash {
    
    pub fn invoke(&self, p: &(*const u8,*const u8)) -> usize {
        
        todo!();
        /*
            // use hash<string> because hash<const char*> would hash pointers and not pointed-to strings
          return hash<string>()(p.first) ^ (~ hash<string>()(p.second));
        */
    }
}

pub fn is_custom_op(op_name: &OperatorName) -> bool {
    
    todo!();
        /*
            static unordered_set<pair<const char*, const char*>, OpNameHash, OpNameEquals> ops {
        ${aten_ops}
        {"", ""}
      };
      return ops.count(make_pair(
                 opName.name.c_str(), opName.overload_name.c_str())) == 0;
        */
}
