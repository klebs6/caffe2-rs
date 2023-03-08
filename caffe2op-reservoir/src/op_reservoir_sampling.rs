crate::ix!();

/**
  | Collect `DATA` tensor into `RESERVOIR`
  | of size `num_to_collect`. `DATA` is
  | assumed to be a batch.
  | 
  | In case where 'objects' may be repeated
  | in data and you only want at most one instance
  | of each 'object' in the reservoir, `OBJECT_ID`
  | can be given for deduplication. If `OBJECT_ID`
  | is given, then you also need to supply
  | additional book-keeping tensors.
  | See input blob documentation for details.
  | 
  | This operator is thread-safe.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReservoirSamplingOp<Context> {
    storage: OperatorStorage,
    context: Context,

    /// number of tensors to collect
    num_to_collect:  i32,
}

register_cpu_operator!{ReservoirSampling, ReservoirSamplingOp<CPUContext>}

num_inputs!{ReservoirSampling, (4,7)}

num_outputs!{ReservoirSampling, (2,4)}

num_inputs_outputs!{ReservoirSampling, 
    |input: i32, output: i32| {
        input / 3 == output / 2
    }
}

inputs!{ReservoirSampling, 
    0 => ("RESERVOIR",            "The reservoir; should be initialized to empty tensor"),
    1 => ("NUM_VISITED",          "Number of examples seen so far; should be initialized to 0"),
    2 => ("DATA",                 "Tensor to collect from. The first dimension is assumed to be batch size. If the object to be collected is represented by multiple tensors, use `PackRecords` to pack them into single tensor."),
    3 => ("MUTEX",                "Mutex to prevent data race"),
    4 => ("OBJECT_ID",            "(Optional, int64) If provided, used for deduplicating object in the reservoir"),
    5 => ("OBJECT_TO_POS_MAP_IN", "(Optional) Auxiliary bookkeeping map. This should be created from  `CreateMap` with keys of type int64 and values of type int32"),
    6 => ("POS_TO_OBJECT_IN",     "(Optional) Tensor of type int64 used for bookkeeping in deduplication")
}

outputs!{ReservoirSampling, 
    0 => ("RESERVOIR",           "Same as the input"),
    1 => ("NUM_VISITED",         "Same as the input"),
    2 => ("OBJECT_TO_POS_MAP",   "(Optional) Same as the input"),
    3 => ("POS_TO_OBJECT",       "(Optional) Same as the input")
}

args!{ReservoirSampling, 
    0 => ("num_to_collect", "The number of random samples to append for each positive samples")
}

enforce_inplace!{ReservoirSampling, vec![(0, 0), (1, 1), (5, 2), (6, 3)]}

should_not_do_gradient!{ReservoirSampling}

input_tags!{
    ReservoirSamplingOp {
        ReservoirIn,
        NumVisitedIn,
        Data,
        Mutex,
        ObjectId,
        ObjectToPosMapIn,
        PosToObjectIn
    }
}

output_tags!{
    ReservoirSamplingOp {
        Reservoir,
        NumVisited,
        ObjectToPosMap,
        PosToObject
    }
}


impl<Context> ReservoirSamplingOp<Context> {
    
    pub fn new(operator_def: OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<Context>(operator_def, ws),
            numToCollect_( OperatorStorage::GetSingleArgument<int>("num_to_collect", -1)) 
        CAFFE_ENFORCE(numToCollect_ > 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& mutex = OperatorStorage::Input<std::unique_ptr<std::mutex>>(MUTEX);
        std::lock_guard<std::mutex> guard(*mutex);

        auto* output = Output(RESERVOIR);
        const auto& input = Input(DATA);

        CAFFE_ENFORCE_GE(input.dim(), 1);

        bool output_initialized = output->numel() > 0 &&
            (static_cast<std::shared_ptr<std::vector<TensorCPU>>*>(
                 output->raw_mutable_data(input.dtype()))[0] != nullptr);

        if (output_initialized) {
          CAFFE_ENFORCE_EQ(output->dim(), input.dim());
          for (size_t i = 1; i < input.dim(); ++i) {
            CAFFE_ENFORCE_EQ(output->size(i), input.size(i));
          }
        }

        auto num_entries = input.sizes()[0];

        if (!output_initialized) {
          // IMPORTANT: Force the output to have the right type before reserving,
          // so that the output gets the right capacity
          auto dims = input.sizes().vec();
          dims[0] = 0;
          output->Resize(dims);
          output->raw_mutable_data(input.dtype());
          output->ReserveSpace(numToCollect_);
        }

        auto* pos_to_object =
            OutputSize() > POS_TO_OBJECT ? Output(POS_TO_OBJECT) : nullptr;
        if (pos_to_object) {
          if (!output_initialized) {
            // Cleaning up in case the reservoir got reset.
            pos_to_object->Resize(0);
            pos_to_object->template mutable_data<int64_t>();
            pos_to_object->ReserveSpace(numToCollect_);
          }
        }

        auto* object_to_pos_map = OutputSize() > OBJECT_TO_POS_MAP
            ? OperatorStorage::Output<MapType64To32>(OBJECT_TO_POS_MAP)
            : nullptr;

        if (object_to_pos_map && !output_initialized) {
          object_to_pos_map->clear();
        }

        auto* num_visited_tensor = Output(NUM_VISITED);
        CAFFE_ENFORCE_EQ(1, num_visited_tensor->numel());
        auto* num_visited = num_visited_tensor->template mutable_data<int64_t>();
        if (!output_initialized) {
          *num_visited = 0;
        }
        CAFFE_ENFORCE_GE(*num_visited, 0);

        if (num_entries == 0) {
          if (!output_initialized) {
            // Get both shape and meta
            output->CopyFrom(input, /* async */ true);
          }
          return true;
        }

        const int64_t* object_id_data = nullptr;
        std::set<int64_t> unique_object_ids;
        if (InputSize() > OBJECT_ID) {
          const auto& object_id = Input(OBJECT_ID);
          CAFFE_ENFORCE_EQ(object_id.dim(), 1);
          CAFFE_ENFORCE_EQ(object_id.numel(), num_entries);
          object_id_data = object_id.template data<int64_t>();
          unique_object_ids.insert(
              object_id_data, object_id_data + object_id.numel());
        }

        const auto num_new_entries = countNewEntries(unique_object_ids);
        auto num_to_copy = std::min<int32_t>(num_new_entries, numToCollect_);
        auto output_batch_size = output_initialized ? output->size(0) : 0;
        auto output_num =
            std::min<size_t>(numToCollect_, output_batch_size + num_to_copy);
        // output_num is >= output_batch_size
        output->ExtendTo(output_num, 50);
        if (pos_to_object) {
          pos_to_object->ExtendTo(output_num, 50);
          // ExtendTo doesn't zero-initialize tensors any more, explicitly clear
          // the memory
          memset(
              pos_to_object->template mutable_data<int64_t>() +
                  output_batch_size * sizeof(int64_t),
              0,
              (output_num - output_batch_size) * sizeof(int64_t));
        }

        auto* output_data =
            static_cast<char*>(output->raw_mutable_data(input.dtype()));
        auto* pos_to_object_data = pos_to_object
            ? pos_to_object->template mutable_data<int64_t>()
            : nullptr;

        auto block_size = input.size_from_dim(1);
        auto block_bytesize = block_size * input.itemsize();
        const auto* input_data = static_cast<const char*>(input.raw_data());

        const auto start_num_visited = *num_visited;

        std::set<int64_t> eligible_object_ids;
        if (object_to_pos_map) {
          for (auto oid : unique_object_ids) {
            if (!object_to_pos_map->count(oid)) {
              eligible_object_ids.insert(oid);
            }
          }
        }

        for (int i = 0; i < num_entries; ++i) {
          if (object_id_data && object_to_pos_map &&
              !eligible_object_ids.count(object_id_data[i])) {
            // Already in the pool or processed
            continue;
          }
          if (object_id_data) {
            eligible_object_ids.erase(object_id_data[i]);
          }
          int64_t pos = -1;
          if (*num_visited < numToCollect_) {
            // append
            pos = *num_visited;
          } else {
            // uniform between [0, num_visited]
            at::uniform_int_from_to_distribution<int64_t> uniformDist(*num_visited+1, 0);
            pos = uniformDist(context_.RandGenerator());
            if (pos >= numToCollect_) {
              // discard
              pos = -1;
            }
          }

          if (pos < 0) {
            // discard
            CAFFE_ENFORCE_GE(*num_visited, numToCollect_);
          } else {
            // replace
            context_.CopyItemsSameDevice(
                input.dtype(),
                block_size,
                input_data + i * block_bytesize,
                output_data + pos * block_bytesize);

            if (object_id_data && pos_to_object_data && object_to_pos_map) {
              auto old_oid = pos_to_object_data[pos];
              auto new_oid = object_id_data[i];
              pos_to_object_data[pos] = new_oid;
              object_to_pos_map->erase(old_oid);
              object_to_pos_map->emplace(new_oid, pos);
            }
          }

          ++(*num_visited);
        }
        // Sanity check
        CAFFE_ENFORCE_EQ(*num_visited, start_num_visited + num_new_entries);
        return true;
        */
    }
    
    #[inline] pub fn count_new_entries(&mut self, unique_object_ids: &HashSet<i64>) -> i32 {
        
        todo!();
        /*
            const auto& input = Input(DATA);
        if (InputSize() <= OBJECT_ID) {
          return input.size(0);
        }
        const auto& object_to_pos_map =
            OperatorStorage::Input<MapType64To32>(OBJECT_TO_POS_MAP_IN);
        return std::count_if(
            unique_object_ids.begin(),
            unique_object_ids.end(),
            [&object_to_pos_map](int64_t oid) {
              return !object_to_pos_map.count(oid);
            });
        */
    }
}
