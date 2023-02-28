/*!
 PyTorch containers are a special zip archive with the following layout
 archive_name.zip contains:
    archive_name/
        version # a file with a single decimal number written in ascii,
                # used to establish the version of the archive format
        model.json # overall model description, this is a json output of
                   # ModelDef from torch.proto
        # the following names are by convention only, model.json will
        # refer to these files by full names
        tensors/
          0 # flat storage for tensor data, meta-data about shapes, etc. is
            # in model.json
          1
          ...
        # code entries will only exist for modules that have methods attached
        code/
          archive_name.py # serialized torch script code (python syntax, using
          PythonPrint) archive_name_my_submodule.py # submodules have separate
          files

 The PyTorchStreamWriter also ensures additional useful properties for these
 files
 1. All files are stored uncompressed.
 2. All files in the archive are aligned to 64 byte boundaries such that
    it is possible to mmap the entire file and get an aligned pointer to
    tensor data.
 3. We universally write in ZIP64 format for consistency.

 The PyTorchStreamReader also provides additional properties:
 1. It can read zip files that are created with common
    zip tools. This means that even though our writer doesn't compress files,
    the reader can still read files that were compressed.
 2. It provides a getRecordOffset function which returns the offset into the
    raw file where file data lives. If the file was written with
    PyTorchStreamWriter it is guaranteed to be 64 byte aligned.

 PyTorchReader/Writer handle checking the version number on the archive format
 and ensure that all files are written to a archive_name directory so they
 unzip cleanly.

 When developing this format we want to pay particular attention to the
 following use cases:

 -- Reading --
 1) Reading with full random access
   a) Reading with file api's such as fread()
   b) mmaping the file and jumping around the mapped region
 2) Reading with 1-pass sequential access
      -> A reader will need to build up a data structure of parsed structures
         as it reads

 -- Writing --
 1) Writing with full random access
 2) Writing with 1-pass sequential access
      -> We must take care not to require updating values that have already
         been written. We place the variable-length index at the end and do
         not put any indicies into the header to fulfill this constraint.

 The model.json, which contains all the metadata information,
 should be written as the last file. One reason is that the size of tensor
 data is usually stable. As long as the shape and type of the tensor do not
 change, the size of the data won't change. On the other sied, the size of the
 serialized model is likely to change, so we store it as the last record, and
 we don't need to move previous records when updating the model data.

 The zip format is sufficiently flexible to handle the above use-case.
 it puts its central directory at the end of the archive and we write
 model.json as the last file when writing after we have accumulated all
 other information.
 */

crate::ix!();

pub struct PyTorchStreamReader<R: ReadAdapterInterface> {
    ar:                       Box<ZipArchive>,
    archive_name:             String,
    archive_name_plus_slash:  String,
    in_:                      Arc<R>,
    version:                  i64,
    reader_lock:              parking_lot::RawMutex,
}

impl<R: ReadAdapterInterface> PyTorchStreamReader<R> {
    
    #[inline] pub fn version(&self) -> u64 {
        
        todo!();
        /*
            return version_;
        */
    }
    
    #[inline] pub fn has_record(&mut self, name: &String) -> bool {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(reader_lock_);
      std::string ss = archive_name_plus_slash_ + name;
      mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
      bool result = ar_->m_last_error != MZ_ZIP_FILE_NOT_FOUND;
      if (!result) {
        ar_->m_last_error = MZ_ZIP_NO_ERROR;
      }
      valid("attempting to locate file ", name.c_str());
      return result;
        */
    }
    
    #[inline] pub fn read(&mut self, 
        pos: u64,
        buf: *mut u8,
        n:   usize) -> usize {
        
        todo!();
        /*
            return in_->read(pos, buf, n, "reading file");
        */
    }
    
    pub fn new(file_name: &String) -> Self {
    
        todo!();
        /*
            : ar_(std::make_unique<mz_zip_archive>()),
          in_(std::make_unique<FileAdapter>(file_name)) 

      init();
        */
    }
    
    pub fn new_from_buf_reader(input: *mut BufReader<R>) -> Self {
    
        todo!();
        /*
            : ar_(std::make_unique<mz_zip_archive>()),
          in_(std::make_unique<IStreamAdapter>(in)) 

      init();
        */
    }
    
    pub fn new_from_read_adapter<T: ReadAdapterInterface>(input: Arc<T>) -> Self {
    
        todo!();
        /*
            : ar_(std::make_unique<mz_zip_archive>()), in_(std::move(in)) 
      init();
        */
    }
    
    #[inline] pub fn init(&mut self)  {
        
        todo!();
        /*
            AT_ASSERT(in_ != nullptr);
      AT_ASSERT(ar_ != nullptr);
      memset(ar_.get(), 0, sizeof(mz_zip_archive));

      size_t size = in_->size();

      // check for the old magic number,
      constexpr size_t kMagicValueLength = 8;
      if (size > kMagicValueLength) {
        char buf[kMagicValueLength];
        read(0, buf, kMagicValueLength);
        valid("checking magic number");
        AT_ASSERTM(
            memcmp("PYTORCH1", buf, kMagicValueLength) != 0,
            "File is an unsupported archive format from the preview release.");
      }

      ar_->m_pIO_opaque = this;
      ar_->m_pRead = istream_read_func;

      mz_zip_reader_init(ar_.get(), size, 0);
      valid("reading zip archive");

      // figure out the archive_name (i.e. the zip folder all the other files are in)
      // all lookups to getRecord will be prefixed by this folder
      int n = mz_zip_reader_get_num_files(ar_.get());
      if (n == 0) {
        CAFFE_THROW("archive does not contain any files");
      }
      size_t name_size = mz_zip_reader_get_filename(ar_.get(), 0, nullptr, 0);
      valid("getting filename");
      std::string buf(name_size, '\0');
      mz_zip_reader_get_filename(ar_.get(), 0, &buf[0], name_size);
      valid("getting filename");
      auto pos = buf.find_first_of('/');
      if (pos == std::string::npos) {
        CAFFE_THROW("file in archive is not in a subdirectory: ", buf);
      }
      archive_name_ = buf.substr(0, pos);
      archive_name_plus_slash_ = archive_name_ + "/";

      // version check
      at::DataPtr version_ptr;
      size_t version_size;
      if (hasRecord(".data/version")) {
        std::tie(version_ptr, version_size) = getRecord(".data/version");
      } else {
        TORCH_CHECK(hasRecord("version"))
        std::tie(version_ptr, version_size) = getRecord("version");
      }
      std::string version(static_cast<const char*>(version_ptr.get()), version_size);
      version_ = caffe2::stoull(version);
      AT_ASSERTM(
          version_ >= kMinSupportedFileFormatVersion,
          "Attempted to read a PyTorch file with version ",
          c10::to_string(version_),
          ", but the minimum supported version for reading is ",
          c10::to_string(kMinSupportedFileFormatVersion),
          ". Your PyTorch script module file is too old. Please re-export it again.");
      AT_ASSERTM(
          version_ <= kMaxSupportedFileFormatVersion,
          "Attempted to read a PyTorch file with version ",
          version_,
          ", but the maximum supported version for reading is ",
          kMaxSupportedFileFormatVersion,
          ". Your PyTorch installation may be too old.");
        */
    }
    
    #[inline] pub fn valid(&mut self, what: *const u8, info: *const u8)  {
        
        todo!();
        /*
            auto err = mz_zip_get_last_error(ar_.get());
      if (err != MZ_ZIP_NO_ERROR) {
        CAFFE_THROW(
            "PytorchStreamReader failed ",
            what,
            info,
            ": ",
            mz_zip_get_error_string(err));
      }
        */
    }
    
    #[inline] pub fn get_all_records(&mut self) -> Vec<String> {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(reader_lock_);
      mz_uint num_files = mz_zip_reader_get_num_files(ar_.get());
      std::vector<std::string> out;
      char buf[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
      for (size_t i = 0; i < num_files; i++) {
        mz_zip_reader_get_filename(ar_.get(), i, buf, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE);
        if (strncmp(
                buf,
                archive_name_plus_slash_.data(),
                archive_name_plus_slash_.size()) != 0) {
          CAFFE_THROW(
              "file in archive is not in a subdirectory ",
              archive_name_plus_slash_,
              ": ",
              buf);
        }
        out.push_back(buf + archive_name_plus_slash_.size());
      }
      return out;
        */
    }
    
    #[inline] pub fn get_recordid(&mut self, name: &String) -> usize {
        
        todo!();
        /*
            std::string ss = archive_name_plus_slash_ + name;
      size_t result = mz_zip_reader_locate_file(ar_.get(), ss.c_str(), nullptr, 0);
      if (ar_->m_last_error == MZ_ZIP_FILE_NOT_FOUND) {
        CAFFE_THROW("file not found: ", ss);
      }
      valid("locating file ", name.c_str());
      return result;
        */
    }
    
    /// return dataptr, size
    #[inline] pub fn get_record(&mut self, name: &String) -> (DataPtr,usize) {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(reader_lock_);
      size_t key = getRecordID(name);
      mz_zip_archive_file_stat stat;
      mz_zip_reader_file_stat(ar_.get(), key, &stat);
      valid("retrieving file meta-data for ", name.c_str());
      at::DataPtr retval = c10::GetCPUAllocator()->allocate(stat.m_uncomp_size);
      mz_zip_reader_extract_to_mem(ar_.get(), key, retval.get(), stat.m_uncomp_size, 0);
      valid("reading file ", name.c_str());

      return std::make_tuple(std::move(retval), stat.m_uncomp_size);
        */
    }
    
    #[inline] pub fn get_record_offset(&mut self, name: &String) -> usize {
        
        todo!();
        /*
            std::lock_guard<std::mutex> guard(reader_lock_);
      mz_zip_archive_file_stat stat;
      mz_zip_reader_file_stat(ar_.get(), getRecordID(name), &stat);
      valid("retrieving file meta-data for ", name.c_str());
      uint8_t local_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
      in_->read(
          stat.m_local_header_ofs,
          local_header,
          MZ_ZIP_LOCAL_DIR_HEADER_SIZE,
          "reading file header");
      size_t filename_len = read_le_16(local_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
      size_t extra_len = read_le_16(local_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
      return stat.m_local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_len + extra_len;
        */
    }
}


impl<R: ReadAdapterInterface> Drop for PyTorchStreamReader<R> {
    fn drop(&mut self) {
        todo!();
        /* 
      mz_zip_clear_last_error(ar_.get());
      mz_zip_reader_end(ar_.get());
      valid("closing reader for archive ", archive_name_.c_str());
 */
    }
}

///---------------------------------------------
pub struct PyTorchStreamWriter<W: Write> {

    current_pos:              usize, // default = 0
    files_written:            Vec<String>,
    ar:                       Box<ZipArchive>,
    archive_name:             String,
    archive_name_plus_slash:  String,
    padding:                  String,
    file_stream:              BufWriter<W>,
    writer_func:              fn(_u0: *const c_void, _u1: usize) -> usize,
    version:                  u64,  // default = kProducedFileFormatVersion
    finalized:                bool, // default = false
    err_seen:                 bool, // default = false
}

// Writer-specific constants
pub const kFieldAlignment: u64 = 64;

#[inline] pub fn istream_read_func(
    p_opaque: *mut c_void,
    file_ofs: u64,
    p_buf:    *mut c_void,
    n:        usize) -> usize {
    
    todo!();
    /*
        auto self = static_cast<PyTorchStreamReader*>(pOpaque);
      return self->read(file_ofs, static_cast<char*>(pBuf), n);
    */
}

#[inline] pub fn basename(name: &String) -> String {
    
    todo!();
    /*
        size_t start = 0;
      for(size_t i = 0; i < name.size(); ++i) {
        if (name[i] == '\\' || name[i] == '/') {
          start = i + 1;
        }
      }

      if (start >= name.size())
        return "";

      size_t end = name.size();
      for(size_t i = end; i > start; --i) {
        if (name[i - 1] == '.') {
          end = i - 1;
          break;
        }
      }
      return name.substr(start, end - start);
    */
}

pub const MZ_ZIP_LOCAL_DIR_HEADER_SIZE: i32 = 30;
pub const MZ_ZIP_LDH_FILENAME_LEN_OFS:  i32 = 26;
pub const MZ_ZIP_LDH_EXTRA_LEN_OFS:     i32 = 28;

/// Returns a record to be appended to the local user extra 
/// data entry in order to make data beginning aligned at 
/// kFieldAlignment bytes boundary.
#[inline] pub fn get_padding(
    cursor:        usize,
    filename_size: usize,
    size:          usize,
    padding_buf:   &mut String) -> usize {
    
    todo!();
    /*
        size_t start = cursor + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + filename_size +
          sizeof(mz_uint16) * 2;
      if (size >= MZ_UINT32_MAX || cursor >= MZ_UINT32_MAX) {
        start += sizeof(mz_uint16) * 2;
        if (size >= MZ_UINT32_MAX) {
          start += 2 * sizeof(mz_uint64);
        }
        if (cursor >= MZ_UINT32_MAX) {
          start += sizeof(mz_uint64);
        }
      }
      size_t mod = start % kFieldAlignment;
      size_t next_offset = (mod == 0) ? start : (start + kFieldAlignment - mod);
      size_t padding_size = next_offset - start;
      size_t padding_size_plus_fbxx = padding_size + 4;
      if (padding_buf.size() < padding_size_plus_fbxx) {
        padding_buf.append(padding_size_plus_fbxx - padding_buf.size(), 'Z');
      }
      // zip extra encoding (key, size_of_extra_bytes)
      padding_buf[0] = 'F';
      padding_buf[1] = 'B';
      padding_buf[2] = (uint8_t)padding_size;
      padding_buf[3] = (uint8_t)(padding_size >> 8);
      return padding_size_plus_fbxx;
    */
}

#[inline] pub fn read_le_16(buf: *mut u8) -> i64 {
    
    todo!();
    /*
        return buf[0] + (buf[1] << 8);
    */
}

#[inline] pub fn ostream_write_func(
    p_opaque: *mut c_void,
    file_ofs: u64,
    p_buf:    *const c_void,
    n:        usize) -> usize {
    
    todo!();
    /*
        auto self = static_cast<PyTorchStreamWriter*>(pOpaque);
      if (self->current_pos_ != file_ofs) {
        CAFFE_THROW("unexpected pos ", self->current_pos_, " vs ", file_ofs);
      }
      size_t ret = self->writer_func_(pBuf, n);
      if (n != ret) {
        self->err_seen_ = true;
      }
      self->current_pos_ += ret;
      return ret;
    */
}

impl<W: Write> PyTorchStreamWriter<W> {
    
    #[inline] pub fn finalized(&self) -> bool {
        
        todo!();
        /*
            return finalized_;
        */
    }
    
    #[inline] pub fn archive_name(&mut self) -> &String {
        
        todo!();
        /*
            return archive_name_;
        */
    }
    
    pub fn new(writer_func: &fn(_u0: *const c_void, _u1: usize) -> usize) -> Self {
    
        todo!();
        /*
            : archive_name_("archive"),
          writer_func_(writer_func) 

      setup(archive_name_);
        */
    }
    
    pub fn new_from_filename(file_name: String) -> Self {
    
        todo!();
        /*
            : archive_name_(basename(file_name)) 

      setup(file_name);
        */
    }
    
    #[inline] pub fn setup(&mut self, file_name: &String)  {
        
        todo!();
        /*
            ar_ = std::make_unique<mz_zip_archive>();
      memset(ar_.get(), 0, sizeof(mz_zip_archive));
      archive_name_plus_slash_ = archive_name_ + "/"; // for writeRecord().

      if (archive_name_.size() == 0) {
        CAFFE_THROW("invalid file name: ", file_name);
      }
      if (!writer_func_) {
        file_stream_.open(
            file_name,
            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        valid("opening archive ", file_name.c_str());
        TORCH_CHECK(file_stream_, "File ", file_name, " cannot be opened.");
        writer_func_ = [this](const void* buf, size_t nbytes) -> size_t {
          file_stream_.write(static_cast<const char*>(buf), nbytes);
          return !file_stream_ ? 0 : nbytes;
        };
      }

      ar_->m_pIO_opaque = this;
      ar_->m_pWrite = ostream_write_func;

      mz_zip_writer_init_v2(ar_.get(), 0, MZ_ZIP_FLAG_WRITE_ZIP64);
      valid("initializing archive ", file_name.c_str());
        */
    }
    
    #[inline] pub fn set_min_version(&mut self, version: u64)  {
        
        todo!();
        /*
            version_ = std::max(version, version_);
        */
    }
    
    #[inline] pub fn write_record(&mut self, 
        name:     &String,
        data:     *const c_void,
        size:     usize,
        compress: bool)  {
        
        todo!();
        /*
            AT_ASSERT(!finalized_);
      AT_ASSERT(!archive_name_plus_slash_.empty());
      std::string full_name = archive_name_plus_slash_ + name;
      size_t padding_size =
          detail::getPadding(ar_->m_archive_size, full_name.size(), size, padding_);
      uint32_t flags = compress ? MZ_BEST_COMPRESSION : 0;
      mz_zip_writer_add_mem_ex_v2(
          ar_.get(),
          full_name.c_str(),
          data,
          size,
          nullptr,
          0,
          flags,
          0,
          0,
          nullptr,
          padding_.c_str(),
          padding_size,
          nullptr,
          0);
      valid("writing file ", name.c_str());
      files_written.push_back(name);
        */
    }
    
    #[inline] pub fn write_end_of_file(&mut self)  {
        
        todo!();
        /*
            // Rewrites version info
      std::string version = c10::to_string(version_);
      version.push_back('\n');
      if (version_ >= 0x6L) {
        writeRecord(".data/version", version.c_str(), version.size());
      } else {
        writeRecord("version", version.c_str(), version.size());
      }

      AT_ASSERT(!finalized_);
      finalized_ = true;

      mz_zip_writer_finalize_archive(ar_.get());
      mz_zip_writer_end(ar_.get());
      valid("writing central directory for archive ", archive_name_.c_str());
      if (file_stream_.is_open()) {
        file_stream_.close();
      }
        */
    }
    
    #[inline] pub fn valid(&mut self, what: *const u8, info: *const u8)  {
        
        todo!();
        /*
            auto err = mz_zip_get_last_error(ar_.get());
      if (err != MZ_ZIP_NO_ERROR) {
        CAFFE_THROW(
            "PytorchStreamWriter failed ",
            what,
            info,
            ": ",
            mz_zip_get_error_string(err));
      }
      if (err_seen_) {
        CAFFE_THROW("PytorchStreamWriter failed ", what, info, ".");
      }
        */
    }
    
    #[inline] pub fn get_all_written_records(&mut self) -> &Vec<String> {
        
        todo!();
        /*
            return files_written;
        */
    }
}

impl<W: Write> Drop for PyTorchStreamWriter<W> {
    fn drop(&mut self) {
        todo!();
        /* 
      if (!finalized_) {
        writeEndOfFile();
      }
       */
    }
}
