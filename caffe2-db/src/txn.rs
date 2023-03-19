crate::ix!();

/**
  | An abstract class for the current database
  | transaction while writing.
  |
  */
pub trait Transaction {

    /**
      | Puts the key value pair to the database.
      |
      */
    fn put(&mut self, key: &String, value: &String);

    /**
      | Commits the current writes.
      |
      */
    fn commit(&mut self);
}
