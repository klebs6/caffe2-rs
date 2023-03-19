crate::ix!();

/**
  | An abstract class for the cursor of the
  | database while reading.
  |
  */
pub trait Cursor {

    /**
      | Seek to a specific key (or if the key does
      | not exist, seek to the immediate next).
      | This is optional for dbs, and in default,
      | SupportsSeek() returns false meaning
      | that the db cursor does not support it.
      |
      */
    fn seek(&mut self, key: &String);

    fn supports_seek(&mut self) -> bool {

        todo!();
        /*
           return false;
           */
    }

    /**
      | Seek to the first key in the database.
      |
      */
    fn seek_to_first(&mut self);

    /**
      | Go to the next location in the database.
      |
      */
    fn next(&mut self);

    /**
      | Returns the current key.
      |
      */
    fn key(&mut self) -> String;

    /**
      | Returns the current value.
      |
      */
    fn value(&mut self) -> String;

    /**
      | Returns whether the current location
      | is valid - for example, if we have reached
      | the end of the database, return false.
      |
      */
    fn valid(&mut self) -> bool;
}
