crate::ix!();

/**
  | Database classes are registered by
  | their names so we can do optional dependencies.
  |
  */
declare_registry!{
    Caffe2DBRegistry, 
    DB, 
    String, 
    Mode
}

#[macro_export] macro_rules! register_caffe2_db {
    ($name:ident, $($arg:ident),*) => {
        /*
        C10_REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)
        */
    }
}
