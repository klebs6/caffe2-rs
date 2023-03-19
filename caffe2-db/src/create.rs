crate::ix!();

/**
  | Returns a database object of the given
  | database type, source and mode. The
  | caller takes the ownership of the pointer.
  | If the database type is not supported,
  | a nullptr is returned. The caller is
  | responsible for examining the validity
  | of the pointer.
  |
  */
#[inline] pub fn createDB(
    db_type: &String, 
    source: &String, 
    mode: Mode) -> Box<dyn DB> 
{
    todo!();
    /*
        auto result = Caffe2DBRegistry()->Create(db_type, source, mode);
      VLOG(1) << ((!result) ? "not found db " : "found db ") << db_type;
      return result;
    */
}

/**
  | Returns whether or not a database exists
  | given the database type and path.
  |
  */
#[inline] pub fn dBExists(db_type: &String, full_db_name: &String) -> bool {
    
    todo!();
    /*
        // Warning! We assume that creating a DB throws an exception if the DB
      // does not exist. If the DB constructor does not follow this design
      // pattern,
      // the returned output (the existence tensor) can be wrong.
      try {
        std::unique_ptr<DB> db(
            caffe2::db::CreateDB(db_type, full_db_name, caffe2::db::READ));
        return true;
      } catch (...) {
        return false;
      }
    */
}
