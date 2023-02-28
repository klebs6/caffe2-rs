crate::ix!();

use crate::{
    OperatorStorage,
    FileReader,
    CPUContext,
    BufferedTokenizer,
    TensorProto_DataType,
    TypeMeta
};

/**
  | Read a batch of rows from the given text
  | file reader instance.
  | 
  | Expects the number of fields to be equal
  | to the number of outputs.
  | 
  | Each output is a 1D tensor containing
  | the values for the given field for each
  | row.
  | 
  | When end of file is reached, returns
  | empty tensors.
  |
  */
pub struct TextFileReaderInstance<'a> {

    file_reader:       FileReader<'a>,
    tokenizer:         BufferedTokenizer,
    field_types:       Vec<i32>,
    field_metas:       Vec<TypeMeta>,
    field_byte_sizes:  Vec<usize>,
    rows_read:         usize, // {0};

    /**
      | hack to guarantee thread-safeness of the
      | read op
      |
      | TODO(azzolini): support multi-threaded
      | reading.
      */
    global_mutex:      parking_lot::RawMutex,
}

register_cpu_operator!{TextFileReaderRead, TextFileReaderReadOp}

num_inputs!{TextFileReaderRead, 1}

num_outputs!{TextFileReaderRead, (1,INT_MAX)}

inputs!{TextFileReaderRead, 
    0 => ("handler", "Pointer to an existing TextFileReaderInstance.")
}

args!{TextFileReaderRead, 
    0 => ("batch_size", "Maximum number of rows to read.")
}

no_gradient!{TextFileReaderRead}

impl<'a> TextFileReaderInstance<'a> {
    
    pub fn new(
        delims:     &Vec<u8>,
        escape:     u8,
        filename:   &String,
        num_passes: i32,
        types:      &Vec<i32>) -> Self {

        todo!();
        /*
            : fileReader(filename),
            tokenizer(Tokenizer(delims, escape), &fileReader, numPasses),
            fieldTypes(types) 

            for (const auto dt : fieldTypes) {
                fieldMetas.push_back(
                    DataTypeToTypeMeta(static_cast<TensorProto_DataType>(dt)));
                fieldByteSizes.push_back(fieldMetas.back().itemsize());
            }
        */
    }
}

/**
  | Create a text file reader. Fields are
  | delimited by <TAB>.
  |
  */
pub struct CreateTextFileReaderOp {
    storage:      OperatorStorage,
    context:      CPUContext,

    filename:     String,
    num_passes:   i32,
    field_types:  Vec<i32>,
}

register_cpu_operator!{CreateTextFileReader, CreateTextFileReaderOp}

num_inputs!{CreateTextFileReader, 0}

num_outputs!{CreateTextFileReader, 1}

outputs!{CreateTextFileReader, 
    0 => ("handler", "Pointer to the created TextFileReaderInstance.")
}

args!{CreateTextFileReader, 
    0 => ("filename",    "Path to the file."),
    1 => ("num_passes",  "Number of passes over the file."),
    2 => ("field_types", "List with type of each field. Type enum is found at core.DataType.")
}

scalar_type!{CreateTextFileReader, TensorProto::UNDEFINED}

no_gradient!{CreateTextFileReader}

impl CreateTextFileReaderOp {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            filename_(GetSingleArgument<string>("filename", "")),
            numPasses_(GetSingleArgument<int>("num_passes", 1)),
            fieldTypes_(GetRepeatedArgument<int>("field_types")) 

        CAFFE_ENFORCE(fieldTypes_.size() > 0, "field_types arg must be non-empty");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            *OperatorStorage::Output<std::unique_ptr<TextFileReaderInstance>>(0) =
            std::unique_ptr<TextFileReaderInstance>(new TextFileReaderInstance(
                {'\n', '\t'}, '\0', filename_, numPasses_, fieldTypes_));
        return true;
        */
    }
}

#[inline] pub fn convert(
    dst_type:  TensorProto_DataType,
    src_start: *const u8,
    src_end:   *const u8,
    dst:       *mut c_void)  {
    
    todo!();
    /*
        switch (dst_type) {
        case TensorProto_DataType_STRING: {
          static_cast<std::string*>(dst)->assign(src_start, src_end);
        } break;
        case TensorProto_DataType_FLOAT: {
          // TODO(azzolini): avoid copy, use faster conversion
          std::string str_copy(src_start, src_end);
          const char* src_copy = str_copy.c_str();
          char* src_copy_end;
          float val = strtof(src_copy, &src_copy_end);
          if (src_copy == src_copy_end) {
            throw std::runtime_error("Invalid float: " + str_copy);
          }
          *static_cast<float*>(dst) = val;
        } break;
        default:
          throw std::runtime_error("Unsupported type.");
      }
    */
}

///--------------------------------
pub struct TextFileReaderReadOp {
    storage:    OperatorStorage,
    context:    CPUContext,
    batch_size: i64,
}

impl TextFileReaderReadOp {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(std::forward<Args>(args)...),
            batchSize_(GetSingleArgument<int>("batch_size", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const int numFields = OutputSize();
        CAFFE_ENFORCE(numFields > 0, "Expected at least one output.");

        auto instance =
            OperatorStorage::Input<std::unique_ptr<TextFileReaderInstance>>(0).get();

        CAFFE_ENFORCE(
            instance->fieldTypes.size() == numFields,
            "Invalid number of outputs. Expected " +
                to_string(instance->fieldTypes.size()) + " got " +
                to_string(numFields));

        // char* datas[numFields];
        // MSVC does not allow using const int, so we will need to dynamically allocate
        // it.
        std::vector<char*> datas(numFields);
        for (int i = 0; i < numFields; ++i) {
          Output(i)->Resize(batchSize_);
          datas[i] = (char*)Output(i)->raw_mutable_data(instance->fieldMetas[i]);
        }

        int rowsRead = 0;
        {
          // TODO(azzolini): support multi-threaded reading
          std::lock_guard<std::mutex> guard(instance->globalMutex_);

          bool finished = false;
          Token token;
          while (!finished && (rowsRead < batchSize_)) {
            int field;
            for (field = 0; field < numFields; ++field) {
              finished = !instance->tokenizer.next(token);
              if (finished) {
                CAFFE_ENFORCE(
                    field == 0, "Invalid number of fields at end of file.");
                break;
              }
              CAFFE_ENFORCE(
                  (field == 0 && token.startDelimId == 0) ||
                      (field > 0 && token.startDelimId == 1),
                  "Invalid number of columns at row ",
                  instance->rowsRead + rowsRead + 1);
              const auto& meta = instance->fieldMetas[field];
              char*& data = datas[field];
              convert(
                  (TensorProto_DataType)instance->fieldTypes[field],
                  token.start,
                  token.end,
                  data);
              data += instance->fieldByteSizes[field];
            }
            if (!finished) {
              ++rowsRead;
            }
          }
          instance->rowsRead += rowsRead;
        }

        for (int i = 0; i < numFields; ++i) {
          Output(i)->ShrinkTo(rowsRead);
        }
        return true;
        */
    }
}

caffe_known_type!{std::unique_ptr<TextFileReaderInstance>}

