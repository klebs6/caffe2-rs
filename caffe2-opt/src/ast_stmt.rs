crate::ix!();

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


