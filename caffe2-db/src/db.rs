crate::ix!();

/**
  | The mode of the database, whether we
  | are doing a read, write, or creating
  | a new database.
  |
  */
pub enum Mode { 
    READ, 
    WRITE, 
    NEW 
}

/**
  | An abstract class for accessing a database
  | of key-value pairs.
  |
  */
pub trait DB {

    /**
      | Closes the database.
      |
      */
    fn close(&mut self);

    /**
      | Returns a cursor to read the database.
      | The caller takes the ownership of the
      | pointer.
      |
      */
    fn new_cursor(&mut self) -> Box<dyn Cursor>;

    /**
      | Returns a transaction to write data
      | to the database. The caller takes the
      | ownership of the pointer.
      |
      */
    fn new_transaction(&mut self) -> Box<dyn Transaction>;
}
