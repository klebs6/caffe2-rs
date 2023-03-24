crate::ix!();

pub trait AnnotateEngine {

    #[inline] fn annotate_engine(&mut self, engine: &String)  {
        
        todo!();
        /*
            engine_ = engine;
        */
    }
}

pub trait GetEngine {

    #[inline] fn engine(&self) -> &String {
        
        todo!();
        /*
            return engine_;
        */
    }
}
