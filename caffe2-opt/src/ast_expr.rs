crate::ix!();

pub struct ASTExpr {
    name:              String, // = "";
    children:          Vec<*mut ASTExpr>,
    is_call_flag:      bool, // default = false
    star_inputs_flag:  bool, // default = false
}

impl Drop for ASTExpr {
    fn drop(&mut self) {
        todo!();
        /* 
        for (ASTExpr* e : children)
          delete e;
       */
    }
}

impl ASTExpr {
    
    #[inline] pub fn is_call(&self) -> bool {
        
        todo!();
        /*
            return isCallFlag;
        */
    }
    
    #[inline] pub fn star_inputs(&self) -> bool {
        
        todo!();
        /*
            return starInputsFlag;
        */
    }
    
    #[inline] pub fn dump(
        &self, 
        level: Option<i32>)
    {
        let level: i32 = level.unwrap_or(0);

        todo!();
        /*
            for (int i = 0; i < level; i++)
          std::cout << "  ";
        if (!isCall())
          std::cout << "Var: " << name << std::endl;
        else {
          std::cout << "Function: " << name << ", args: " << std::endl;
          for (auto* e : children) {
            e->dump(level + 1);
          }
        }
        */
    }
}

