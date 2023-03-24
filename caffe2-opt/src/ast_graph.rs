crate::ix!();

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


