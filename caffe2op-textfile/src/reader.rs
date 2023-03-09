crate::ix!();

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
