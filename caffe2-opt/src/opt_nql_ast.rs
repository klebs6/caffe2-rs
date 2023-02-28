crate::ix!();

///----------------------------------
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

///----------------------------------
pub struct ASTStmt {
    lhs:  Vec<String>,
    rhs:  *mut ASTExpr, // default = NULL
}

impl Drop for ASTStmt {
    fn drop(&mut self) {
        todo!();
        /* 
        delete rhs;
       */
    }
}

impl ASTStmt {
    
    #[inline] fn dump(
        &self, 
        level: Option<i32>)
    {

        let level: i32 = level.unwrap_or(0);

        todo!();
        /*
            for (int i = 0; i < level; i++)
          std::cout << "  ";
        std::cout << "LHS:" << std::endl;
        for (auto s : lhs) {
          for (int i = 0; i < level + 1; i++)
            std::cout << "  ";
          std::cout << s << std::endl;
        }
        rhs->dump(level);
        */
    }
}

///----------------------------------
pub struct ASTGraph {

    name:   String,
    stmts:  Vec<ASTStmt>,

}

impl Drop for ASTGraph {
    fn drop(&mut self) {
        todo!();
        /* 
        for (auto s : stmts)
          delete s;
       */
    }
}

impl ASTGraph {
    
    #[inline] pub fn dump(&self)  {
        
        todo!();
        /*
            std::cout << "GRAPH: " << name << std::endl;
        for (auto s : stmts)
          s->dump(1);
        */
    }
}

#[inline] pub fn alloc_string() -> *mut String {
    
    todo!();
    /*
    
    */
}


#[inline] pub fn alloc_vector() -> *mut Vec<*mut c_void> {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn parse_string(c: *const u8, g: *mut ASTGraph)  {
    
    todo!();
    /*
    
    */
}

#[inline] pub fn parse_file(c: *const u8, g: *mut ASTGraph)  {
    
    todo!();
    /*
    
    */
}
