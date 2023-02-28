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

///-------------------------
pub struct DBReaderSerializer {
    base: dyn BlobSerializerBase,
}

///-------------------------
pub struct DBReaderDeserializer {
    base: dyn BlobDeserializerBase,
}

caffe_known_type![db::DBReader];
caffe_known_type![db::Cursor];

/**
  | Below, we provide a bare minimum database
  | "minidb" as a reference implementation as well
  | as a portable choice to store data.
  |
  | Note that the MiniDB classes are not exposed
  | via a header file - they should be created
  | directly via the db interface. See MiniDB for
  | details.
  */
pub struct MiniDBCursor<'a> {
    lock:      MutexGuard<'a, parking_lot::RawMutex>,
    file:      *mut libc::FILE,
    valid:     bool,
    key_len:   i32,
    key:       Vec<u8>,
    value_len: i32,
    value:     Vec<u8>,
}

impl<'a> MiniDBCursor<'a> {

    pub fn new(
        f: *mut libc::FILE, 
        mutex: *mut parking_lot::RawMutex) -> Self 
    {
        todo!();
        /*
            : file_(f), lock_(*mutex), valid_(true) 
        // We call Next() to read in the first entry.
        Next();
        */
    }
}

impl<'a> Cursor for MiniDBCursor<'a> {
    
    #[inline] fn seek(&mut self, key: &String)  {
        
        todo!();
        /*
            LOG(FATAL) << "MiniDB does not support seeking to a specific key.";
        */
    }
    
    #[inline] fn seek_to_first(&mut self)  {
        
        todo!();
        /*
            fseek(file_, 0, SEEK_SET);
        CAFFE_ENFORCE(!feof(file_), "Hmm, empty file?");
        // Read the first item.
        valid_ = true;
        Next();
        */
    }
    
    #[inline] fn next(&mut self)  {
        
        todo!();
        /*
            // First, read in the key and value length.
        if (fread(&key_len_, sizeof(int), 1, file_) == 0) {
          // Reaching EOF.
          VLOG(1) << "EOF reached, setting valid to false";
          valid_ = false;
          return;
        }
        CAFFE_ENFORCE_EQ(fread(&value_len_, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_GT(key_len_, 0);
        CAFFE_ENFORCE_GT(value_len_, 0);
        // Resize if the key and value len is larger than the current one.
        if (key_len_ > (int)key_.size()) {
          key_.resize(key_len_);
        }
        if (value_len_ > (int)value_.size()) {
          value_.resize(value_len_);
        }
        // Actually read in the contents.
        CAFFE_ENFORCE_EQ(
            fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
        CAFFE_ENFORCE_EQ(
            fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
        // Note(Yangqing): as we read the file, the cursor naturally moves to the
        // beginning of the next entry.
        */
    }
    
    #[inline] fn key(&mut self) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
        return string(key_.data(), key_len_);
        */
    }
    
    #[inline] fn value(&mut self) -> String {
        
        todo!();
        /*
            CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
        return string(value_.data(), value_len_);
        */
    }
    
    #[inline] fn valid(&mut self) -> bool {
        
        todo!();
        /*
            return valid_;
        */
    }
}


///---------------------------------
pub struct MiniDBTransaction<'a> {
    file: *mut libc::FILE,
    lock: MutexGuard<'a, parking_lot::RawMutex>,
}

impl<'a> Drop for MiniDBTransaction<'a> {
    fn drop(&mut self) {
        todo!();
        /* 
        Commit();
       */
    }
}

impl<'a> MiniDBTransaction<'a> {

    pub fn new(
        f: *mut libc::FILE, 
        mutex: *mut parking_lot::RawMutex) -> Self 
    {
        todo!();
        /*
            : file_(f), lock_(*mutex)
        */
    }
    
}

impl<'a> Transaction for MiniDBTransaction<'a> {
    
    #[inline] fn put(
        &mut self, 
        key:   &String, 
        value: &String)  
    {
        todo!();
        /*
            int key_len = key.size();
        int value_len = value.size();
        CAFFE_ENFORCE_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
        CAFFE_ENFORCE_EQ(
            fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
        CAFFE_ENFORCE_EQ(
            fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
        */
    }
    
    #[inline] fn commit(&mut self)  {
        
        todo!();
        /*
            if (file_ != nullptr) {
          CAFFE_ENFORCE_EQ(fflush(file_), 0);
          file_ = nullptr;
        }
        */
    }
}

///----------------------------
pub struct MiniDB {

    file: *mut libc::FILE,

    /**
      | access mutex makes sure we don't have
      | multiple cursors/transactions reading
      | the same file.
      |
      */
    file_access_mutex: parking_lot::RawMutex,

    mode: DatabaseMode,
}

impl Drop for MiniDB {
    fn drop(&mut self) {
        todo!();
        /* 
        Close();
       */
    }
}

impl DB for MiniDB {

    #[inline] fn close(&mut self)  {
        
        todo!();
        /*
            if (file_) {
          fclose(file_);
        }
        file_ = nullptr;
        */
    }
    
    #[inline] fn new_cursor(&mut self) -> Box<dyn Cursor> {
        
        todo!();
        /*
            CAFFE_ENFORCE_EQ(this->mode_, READ);
        return make_unique<MiniDBCursor>(file_, &file_access_mutex_);
        */
    }
    
    #[inline] fn new_transaction(&mut self) -> Box<dyn Transaction> {
        
        todo!();
        /*
            CAFFE_ENFORCE(this->mode_ == NEW || this->mode_ == WRITE);
        return make_unique<MiniDBTransaction>(file_, &file_access_mutex_);
        */
    }
}

impl MiniDB {
    
    pub fn new(source: &String, mode: Mode) -> Self {
        todo!();
        /*
            : DB(source, mode), file_(nullptr) 

        switch (mode) {
          case NEW:
            file_ = fopen(source.c_str(), "wb");
            break;
          case WRITE:
            file_ = fopen(source.c_str(), "ab");
            fseek(file_, 0, SEEK_END);
            break;
          case READ:
            file_ = fopen(source.c_str(), "rb");
            break;
        }
        CAFFE_ENFORCE(file_, "Cannot open file: " + source);
        VLOG(1) << "Opened MiniDB " << source;
        */
    }
}

register_caffe2_db![MiniDB, MiniDB];
register_caffe2_db![minidb, MiniDB];

impl BlobSerializerBase for DBReaderSerializer {
    
    /**
      | Serializes a DBReader. Note that this
      | blob has to contain DBReader, otherwise
      | this function produces a fatal error.
      |
      */
    #[inline] fn serialize(
        &mut self, 
        pointer:   *const c_void,
        type_meta: TypeMeta,
        name:      &String,
        acceptor:  SerializationAcceptor,
        options:   Option<&BlobSerializationOptions>)  
    {
        todo!();
        /*
            CAFFE_ENFORCE(typeMeta.Match<DBReader>());
      const auto& reader = *static_cast<const DBReader*>(pointer);
      DBReaderProto proto;
      proto.set_name(name);
      proto.set_source(reader.source_);
      proto.set_db_type(reader.db_type_);
      if (reader.cursor() && reader.cursor()->SupportsSeek()) {
        proto.set_key(reader.cursor()->key());
      }
      BlobProto blob_proto;
      blob_proto.set_name(name);
      blob_proto.set_type("DBReader");
      blob_proto.set_content(SerializeAsString_EnforceCheck(proto));
      acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
        */
    }
}

impl DBReaderDeserializer {
    
    #[inline] pub fn deserialize(
        &mut self, 
        proto: &BlobProto, 
        blob: *mut Blob)  
    {
        todo!();
        /*
            DBReaderProto reader_proto;
      CAFFE_ENFORCE(
          reader_proto.ParseFromString(proto.content()),
          "Cannot parse content into a DBReaderProto.");
      blob->Reset(new DBReader(reader_proto));
        */
    }
}

// Serialize TensorCPU.
register_blob_serializer![
    /*
    (TypeMeta::Id<DBReader>()), 
    DBReaderSerializer
    */
];

register_blob_deserializer![
    /*
    DBReader, 
    DBReaderDeserializer
    */
];
