crate::ix!();

/**
  | A reader wrapper for DB that also allows
  | us to serialize it.
  |
  */
pub struct DBReader {
    db_type:        String,
    source:         String,
    db:             Box<dyn DB>,
    cursor:         Box<dyn Cursor>,
    reader_mutex:   parking_lot::RawMutex,
    num_shards:     u32, // default = 0
    shard_id:       u32, // default = 0
}

impl From<&DBReaderProto> for DBReader {

    fn from(proto: &DBReaderProto) -> Self {
        todo!();
        /*
            Open(proto.db_type(), proto.source());
        if (proto.has_key()) {
          CAFFE_ENFORCE(
              cursor_->SupportsSeek(),
              "Encountering a proto that needs seeking but the db type "
              "does not support it.");
          cursor_->Seek(proto.key());
        }
        num_shards_ = 1;
        shard_id_ = 0;
        */
    }
}

impl From<Box<dyn DB>> for DBReader {

    fn from(db: Box<dyn DB>) -> Self {
        todo!();
        /*
            : db_type_("<memory-type>"),
            source_("<memory-source>"),
            db_(std::move(db)) 

        CAFFE_ENFORCE(db_.get(), "Passed null db");
        cursor_ = db_->NewCursor();
        */
    }
}

impl DBReader {
    
    pub fn new(
        db_type:    &String,
        source:     &String,
        num_shards: i32,
        shard_id:   i32) -> Self 
    {
        todo!();
        /*
            Open(db_type, source, num_shards, shard_id);
        */
    }
    
    #[inline] pub fn open(
        &mut self, 
        db_type:    &String,
        source:     &String,
        num_shards: Option<i32>,
        shard_id:   Option<i32>)  
    {
        let num_shards: i32 = num_shards.unwrap_or(1);
        let shard_id: i32 = shard_id.unwrap_or(0);

        todo!();
        /*
            // Note(jiayq): resetting is needed when we re-open e.g. leveldb where no
        // concurrent access is allowed.
        cursor_.reset();
        db_.reset();
        db_type_ = db_type;
        source_ = source;
        db_ = CreateDB(db_type_, source_, READ);
        CAFFE_ENFORCE(
            db_,
            "Cannot find db implementation of type ",
            db_type,
            " (while trying to open ",
            source_,
            ")");
        InitializeCursor(num_shards, shard_id);
        */
    }
    
    #[inline] pub fn open_with_db(
        &mut self, 
        db:         Box<dyn DB>,
        num_shards: Option<i32>,
        shard_id:   Option<i32>)  
    {
        let num_shards: i32 = num_shards.unwrap_or(1);
        let shard_id: i32 = shard_id.unwrap_or(0);

        todo!();
        /*
            cursor_.reset();
        db_.reset();
        db_ = std::move(db);
        CAFFE_ENFORCE(db_.get(), "Passed null db");
        InitializeCursor(num_shards, shard_id);
        */
    }
    
    /**
      | Read a set of key and value from the db
      | and move to next. Thread safe.
      | 
      | The string objects key and value must
      | be created by the caller and explicitly
      | passed in to this function. This saves
      | one additional object copy.
      | 
      | If the cursor reaches its end, the reader
      | will go back to the head of the db. This
      | function can be used to enable multiple
      | input ops to read the same db.
      | 
      | Note(jiayq): we loosen the definition
      | of a const function here a little bit:
      | the state of the cursor is actually changed.
      | However, this allows us to pass in a DBReader
      | to an Operator without the need of a duplicated
      | output blob.
      |
      */
    #[inline] pub fn read(
        &self, 
        key:   *mut String,
        value: *mut String)
    {

        todo!();
        /*
            CAFFE_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
        std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
        *key = cursor_->key();
        *value = cursor_->value();

        // In sharded mode, each read skips num_shards_ records
        for (uint32_t s = 0; s < num_shards_; s++) {
          cursor_->Next();
          if (!cursor_->Valid()) {
            MoveToBeginning();
            break;
          }
        }
        */
    }
    
    /**
      | -----------
      | @brief
      | 
      | Seeks to the first key. Thread safe.
      |
      */
    #[inline] pub fn seek_to_first(&self)  {
        
        todo!();
        /*
            CAFFE_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
        std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
        MoveToBeginning();
        */
    }

    /**
      | Returns the underlying cursor of the
      | db reader.
      | 
      | -----------
      | @note
      | 
      | if you directly use the cursor, the read
      | will not be thread safe, because there
      | is no mechanism to stop multiple threads
      | from accessing the same cursor. You
      | should consider using Read() explicitly.
      |
      */
    #[inline] pub fn cursor(&self) -> *mut dyn Cursor {
        
        todo!();
        /*
            VLOG(1) << "Usually for a DBReader you should use Read() to be "
                   "thread safe. Consider refactoring your code.";
        return cursor_.get();
        */
    }
    
    #[inline] pub fn initialize_cursor(
        &mut self, 
        num_shards: i32,
        shard_id:   i32)  
    {
        
        todo!();
        /*
            CAFFE_ENFORCE(num_shards >= 1);
        CAFFE_ENFORCE(shard_id >= 0);
        CAFFE_ENFORCE(shard_id < num_shards);
        num_shards_ = num_shards;
        shard_id_ = shard_id;
        cursor_ = db_->NewCursor();
        SeekToFirst();
        */
    }
    
    #[inline] pub fn move_to_beginning(&self)  {
        
        todo!();
        /*
            cursor_->SeekToFirst();
        for (uint32_t s = 0; s < shard_id_; s++) {
          cursor_->Next();
          CAFFE_ENFORCE(
              cursor_->Valid(), "Db has fewer rows than shard id: ", s, shard_id_);
        }
        */
    }
}
