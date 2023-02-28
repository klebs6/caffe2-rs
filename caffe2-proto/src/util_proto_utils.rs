crate::ix!();

//UNKNOWN, TODO
pub struct RepeatedPtr<T>       { phantom: PhantomData<T>, }
pub struct RepeatedPtrField<T>  { phantom: PhantomData<T>, }
pub struct MessageLite          { }

#[cfg(caffe2_use_lite_proto)]
pub mod TextFormat {

    #[inline] pub fn parse_from_string(
        spec:  &String,
        proto: *mut MessageLite) -> bool 
    {
        
        todo!();
        /*
            LOG(FATAL) << "If you are running lite version, you should not be "
                    << "calling any text-format protobuffers.";
                return false;
        */
    }
}

/**
  | Text format MessageLite wrappers: these
  | functions do nothing but just allowing things
  | to compile. It will produce a runtime error if
  | you are using MessageLite but still want text
  | support.
  */
#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn read_proto_lite_from_text_file(
    filename: *const u8,
    proto:    *mut MessageLite) -> bool 
{
    todo!();
    /*
        LOG(FATAL) << "If you are running lite version, you should not be "
                      << "calling any text-format protobuffers.";
      return false;  // Just to suppress compiler warning.
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn read_proto_lite_from_text_file(
    filename: String,
    proto:    *mut MessageLite) -> bool 
{
    todo!();
    /*
        return ReadProtoFromTextFile(filename.c_str(), proto);
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn write_proto_lite_to_text_file(
    proto:          &MessageLite,
    filename:       *const u8,
    throw_if_error: bool)  
{
    let throw_if_error: bool = throw_if_error.unwrap_or(true);

    todo!();
    /*
        LOG(FATAL) << "If you are running lite version, you should not be "
                      << "calling any text-format protobuffers.";
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn write_proto_lite_to_text_file(
    proto:          &MessageLite,
    filename:       &String,
    throw_if_error: bool)  
{
    let throw_if_error: bool = throw_if_error.unwrap_or(true);

    todo!();
    /*
        return WriteProtoToTextFile(proto, filename.c_str(), throwIfError);
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn read_proto_lite_from_file(
    filename: *const u8,
    proto:    *mut MessageLite) -> bool 
{
    todo!();
    /*
        return (ReadProtoFromBinaryFile(filename, proto) ||
              ReadProtoFromTextFile(filename, proto));
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn read_proto_from_filename(
    filename: &String,
    proto:    *mut MessageLite) -> bool 
{
    todo!();
    /*
        return ReadProtoFromFile(filename.c_str(), proto);
    */
}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn read_proto_from_text_filename<M: ::protobuf::Message>(
    filename: String,
    proto:    *mut M) -> bool 
{
    
    todo!();
    /*
        return ReadProtoFromTextFile(filename.c_str(), proto);
    */
}

/**
  | Read Proto from a file, letting the code
  | figure out if it is text or binary.
  |
  */
#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn read_proto_from_file<M: ::protobuf::Message>(
    filename: *const u8,
    proto:    *mut M) -> bool 
{
    
    todo!();
    /*
        return (ReadProtoFromBinaryFile(filename, proto) ||
              ReadProtoFromTextFile(filename, proto));
    */
}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn read_proto_from_filename<M: ::protobuf::Message>(
    filename: &String,
    proto:    *mut M) -> bool 
{
    todo!();
    /*
        return ReadProtoFromFile(filename.c_str(), proto);
    */
}

/**
  | IterableInputs default = std::initializer_list<string>
  | 
  | IterableOutputs default = std::initializer_list<string>
  | 
  | IterableArgs default = std::initializer_list<Argument>
  |
  */
#[inline] pub fn create_operator_def_with_args<IterableInputs, IterableOutputs, IterableArgs>(
    ty:            &String,
    name:          &String,
    inputs:        &IterableInputs,
    outputs:       &IterableOutputs,
    args:          &IterableArgs,
    device_option: &DeviceOption,
    engine:        &String) -> OperatorDef {
    todo!();
    /*
        OperatorDef def;
      def.set_type(type);
      def.set_name(name);
      for (const string& in : inputs) {
        def.add_input(in);
      }
      for (const string& out : outputs) {
        def.add_output(out);
      }
      for (const Argument& arg : args) {
        def.add_arg()->CopyFrom(arg);
      }
      if (device_option.has_device_type()) {
        def.mutable_device_option()->CopyFrom(device_option);
      }
      if (engine.size()) {
        def.set_engine(engine);
      }
      return def;
    */
}

/**
  | A simplified version compared to the
  | full
  | 
  | CreateOperator, if you do not need to
  | specify args.
  | 
  | IterableInputs default = std::initializer_list<string>
  | 
  | IterableOutputs default = std::initializer_list<string>
  |
  */
#[inline] pub fn create_operator_def<IterableInputs, IterableOutputs>(
    ty:            &String,
    name:          &String,
    inputs:        &IterableInputs,
    outputs:       &IterableOutputs,
    device_option: &DeviceOption,
    engine:        &String) -> OperatorDef {
    todo!();
    /*
        return CreateOperatorDef(
          type,
          name,
          inputs,
          outputs,
          std::vector<Argument>(),
          device_option,
          engine);
    */
}

/**
  | -----------
  | @brief
  | 
  | A helper class to index into arguments.
  | 
  | This helper helps us to more easily index
  | into a set of arguments that are present
  | in the operator.
  | 
  | To save memory, the argument helper
  | does not copy the operator def, so one
  | would need to make sure that the lifetime
  | of the OperatorDef object outlives
  | that of the ArgumentHelper.
  |
  */
pub struct ArgumentHelper {
    arg_map: HashMap<String,Argument>,
}

impl ArgumentHelper {
    
    pub fn new(def: &OperatorDef) -> Self {
        todo!();
        /*
            for (auto& arg : def.arg()) {
        if (arg_map_.count(arg.name())) {
          if (arg.SerializeAsString() != arg_map_[arg.name()].SerializeAsString()) {
            // If there are two arguments of the same name but different contents,
            // we will throw an error.
            CAFFE_THROW(
                "Found argument of the same name ",
                arg.name(),
                "but with different contents.",
                ProtoDebugString(def));
          } else {
            LOG(WARNING) << "Duplicated argument name [" << arg.name()
                         << "] found in operator def: " << ProtoDebugString(def);
          }
        }
        arg_map_[arg.name()] = arg;
      }
        */
    }
    
    pub fn new_from_netdef(netdef: &NetDef) -> Self {
        todo!();
        /*
            for (auto& arg : netdef.arg()) {
        CAFFE_ENFORCE(
            arg_map_.count(arg.name()) == 0,
            "Duplicated argument name [",
            arg.name(),
            "] found in net def: ",
            ProtoDebugString(netdef));
        arg_map_[arg.name()] = arg;
      }
        */
    }
    
    #[inline] pub fn has_argument(&self, name: &String) -> bool {
        
        todo!();
        /*
            return arg_map_.count(name);
        */
    }
    
    #[inline] pub fn has_argument_with_def<Def>(def: &Def, name: &String) -> bool {
        todo!();
        /*
            return ArgumentHelper(def).HasArgument(name);
        */
    }
    
    #[inline] pub fn get_single_argument<Def, T>(
        def:           &Def,
        name:          &str,
        default_value: T) -> T {
        todo!();
        /*
            return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
        */
    }
    
    #[inline] pub fn has_single_argument_of_type<Def, T>(def: &Def, name: &String) -> bool {
        todo!();
        /*
            return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
        */
    }
    
    #[inline] pub fn get_repeated_argument<Def, T>(
        def:           &Def,
        name:          &String,
        default_value: &Vec<T>) -> Vec<T> {
        todo!();
        /*
            return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
        */
    }
    
    #[inline] pub fn get_message_argument_with_def<Def, MessageType>(
        def:  &Def, 
        name: &String) -> MessageType 
    {
        todo!();
        /*
            return ArgumentHelper(def).GetMessageArgument<MessageType>(name);
        */
    }
    
    #[inline] pub fn get_repeated_message_argument_with_def<Def, MessageType>(def: &Def, name: &String) -> Vec<MessageType> {
        todo!();
        /*
            return ArgumentHelper(def).GetRepeatedMessageArgument<MessageType>(name);
        */
    }
    
    #[inline] pub fn remove_argument<Def>(def: &mut Def, index: i32) -> bool {
        todo!();
        /*
            if (index >= def.arg_size()) {
          return false;
        }
        if (index < def.arg_size() - 1) {
          def.mutable_arg()->SwapElements(index, def.arg_size() - 1);
        }
        def.mutable_arg()->RemoveLast();
        return true;
        */
    }
    
    #[inline] pub fn get_message_argument<MessageType>(&self, name: &String) -> MessageType {
        todo!();
        /*
            CAFFE_ENFORCE(arg_map_.count(name), "Cannot find parameter named ", name);
        MessageType message;
        if (arg_map_.at(name).has_s()) {
          CAFFE_ENFORCE(
              message.ParseFromString(arg_map_.at(name).s()),
              "Failed to parse content from the string");
        } else {
          VLOG(1) << "Return empty message for parameter " << name;
        }
        return message;
        */
    }
    
    #[inline] pub fn get_repeated_message_argument<MessageType>(&self, name: &String) -> Vec<MessageType> {
        todo!();
        /*
            CAFFE_ENFORCE(arg_map_.count(name), "Cannot find parameter named ", name);
        vector<MessageType> messages(arg_map_.at(name).strings_size());
        for (int i = 0; i < messages.size(); ++i) {
          CAFFE_ENFORCE(
              messages[i].ParseFromString(arg_map_.at(name).strings(i)),
              "Failed to parse content from the string");
        }
        return messages;
        */
    }
}

/* ----------- **** Arguments Utils *****  ----------- */

#[inline] pub fn add_argument<T, Def>(
    name:  &String,
    value: &T,
    def:   *mut Def) 
{
    todo!();
    /*
        GetMutableArgument(name, true, def)->CopyFrom(MakeArgument(name, value));
    */
}

/* --------- **** End Arguments Utils *****  --------- */

/*
impl PartialEq<DeviceOption> for DeviceOption {
    
    #[inline] fn eq(&self, other: &DeviceOption) -> bool {
        todo!();
        /*
            return IsSameDevice(dl, dr);
        */
    }
}
*/

impl Eq for DeviceOption {}

impl Hash for DeviceOption {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
        /*
        typedef caffe2::DeviceOption argument_type;
        typedef std::size_t result_type;
        std::string serialized;
        CAFFE_ENFORCE(device_option.SerializeToString(&serialized));
        return std::hash<std::string>{}(serialized);
        */
    }
}

/**
  | A wrapper function to return device
  | name string for use in blob serialization
  | / deserialization.
  | 
  | This should have one to one correspondence
  | with caffe2/proto/caffe2.proto:
  | enum DeviceType.
  | 
  | -----------
  | @note
  | 
  | we can't use DeviceType_Name, because
  | that is only available in protobuf-full,
  | and some platforms (like mobile) may
  | want to use protobuf-lite instead.
  |
  */
#[inline] pub fn device_type_name(d: &i32) -> String {
    
    todo!();
    /*
        return at::DeviceTypeName(static_cast<at::DeviceType>(d));
    */
}

#[inline] pub fn device_id(option: &DeviceOption) -> i32 {
    
    todo!();
    /*
        switch (option.device_type()) {
        case PROTO_CPU:
          return option.numa_node_id();
        case PROTO_CUDA:
        case PROTO_HIP:
          return option.device_id();
        case PROTO_MKLDNN:
          return option.numa_node_id();
        default:
          CAFFE_THROW("Unknown device id for device type: ", option.device_type());
      }
    */
}

/**
  | Returns if the two DeviceOptions are
  | pointing to the same device.
  |
  */
#[inline] pub fn is_same_device(
    lhs: &DeviceOption,
    rhs: &DeviceOption) -> bool 
{
    todo!();
    /*
        return (
          lhs.device_type() == rhs.device_type() &&
          lhs.device_id() == rhs.device_id() &&
          lhs.node_name() == rhs.node_name() &&
          lhs.numa_node_id() == rhs.numa_node_id());
    */
}

#[inline] pub fn is_cpudevice_type(device_type: i32) -> bool {
    
    todo!();
    /*
        static const std::unordered_set<int> cpu_types{
          PROTO_CPU,
          PROTO_MKLDNN,
          PROTO_IDEEP,
      };
      return cpu_types.count(device_type);
    */
}

#[inline] pub fn is_gpudevice_type(device_type: i32) -> bool {
    
    todo!();
    /*
        static const std::unordered_set<int> gpu_types{
          PROTO_CUDA,
          PROTO_HIP,
      };
      return gpu_types.count(device_type);
    */
}

/**
  | Common interfaces that reads file contents
  | into a string.
  |
  */
#[inline] pub fn read_string_from_file(
    filename: *const u8,
    str_:     *mut String) -> bool 
{
    todo!();
    /*
        std::ifstream ifs(filename, std::ios::in);
      if (!ifs) {
        VLOG(1) << "File cannot be opened: " << filename
                << " error: " << ifs.rdstate();
        return false;
      }
      ifs.seekg(0, std::ios::end);
      size_t n = ifs.tellg();
      str->resize(n);
      ifs.seekg(0);
      ifs.read(&(*str)[0], n);
      return true;
    */
}

#[inline] pub fn write_string_to_file(
    str_:     &String,
    filename: *const u8) -> bool 
{
    todo!();
    /*
        std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
      if (!ofs.is_open()) {
        VLOG(1) << "File cannot be created: " << filename
                << " error: " << ofs.rdstate();
        return false;
      }
      ofs << str;
      return true;
    */
}

/**
  | IO-specific proto functions: we will
  | deal with the protocol buffer lite and
  | full versions differently.
  |
  */

// ------------------------------------------------ [Lite runtime]

#[cfg(caffe2_use_lite_proto)]
pub struct IfstreamInputStream {
    base: CopyingInputStream,
    ifs:  std::fs::File,
}

#[cfg(caffe2_use_lite_proto)]
impl Drop for IfstreamInputStream {
    fn drop(&mut self) {
        todo!();
        /* 
        ifs_.close();
       */
    }
}

#[cfg(caffe2_use_lite_proto)]
impl IfstreamInputStream {
    
    pub fn new(filename: &String) -> Self {
        todo!();
        /*
            : ifs_(filename.c_str(), std::ios::in | std::ios::binary)
        */
    }
    
    #[inline] pub fn read(&mut self, buffer: *mut c_void, size: i32) -> i32 {
        
        todo!();
        /*
            if (!ifs_) {
          return -1;
        }
        ifs_.read(static_cast<char*>(buffer), size);
        return ifs_.gcount();
        */
    }
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn proto_debug_string(proto: &MessageLite) -> String {
    
    todo!();
    /*
        string serialized = proto.SerializeAsString();
      for (char& c : serialized) {
        if (c < 0x20 || c >= 0x7f) {
          c = '?';
        }
      }
      return serialized;
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn parse_proto_from_large_string(
    str_:  &String,
    proto: *mut MessageLite) -> bool 
{
    todo!();
    /*
        ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
      ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
      // Set PlanDef message size limit to 2G.
      coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
      return proto->ParseFromCodedStream(&coded_stream);
    */
}

/**
  | Common interfaces that are supported
  | by both lite and full protobuf.
  |
  */
#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn read_proto_lite_from_binary_file(
    filename: *const u8,
    proto:    *mut MessageLite) -> bool 
{
    todo!();
    /*
        ::google::protobuf::io::CopyingInputStreamAdaptor stream(
          new IfstreamInputStream(filename));
      stream.SetOwnsCopyingStream(true);
      // Total bytes hard limit / warning limit are set to 2GB and 512MB
      // respectively.
      ::google::protobuf::io::CodedInputStream coded_stream(&stream);
      coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
      return proto->ParseFromCodedStream(&coded_stream);
    */
}

#[cfg(caffe2_use_lite_proto)]
#[inline] pub fn write_proto_lite_to_binary_file(
    proto:    &MessageLite,
    filename: *const u8)  
{
    
    todo!();
    /*
        LOG(FATAL) << "Not implemented yet.";
    */
}

#[cfg(not(caffe2_use_lite_proto))]
pub mod TextFormat {

    #[inline] pub fn parse_from_string<M: ::protobuf::Message>(
        spec:  &String,
        proto: *mut M) -> bool 
    {
        
        todo!();
        /*
            string bc_spec = spec;

          {
            auto num_replaced = c10::ReplaceAll(bc_spec, "cuda_gpu_id", "device_id");
            if (num_replaced) {
              LOG(ERROR) << "Your model was serialized in Protobuf TextFormat and "
                         << "it has " << num_replaced
                         << " places using the deprecated field name 'cuda_gpu_id'!\n"
                         << spec
                         << "\nPlease re-export your model in Protobuf binary format "
                         << "to make it backward compatible for field renaming.";
            }
          }

          return ::google::protobuf::TextFormat::ParseFromString(
              std::move(bc_spec), proto);
        */
    }

}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn proto_debug_string<M: ::protobuf::Message>(proto: &M) -> String {
    
    todo!();
    /*
        return proto.ShortDebugString();
    */
}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn parse_proto_from_large_string<M: ::protobuf::Message>(
    str_:  &String,
    proto: *mut M) -> bool {
    
    todo!();
    /*
        ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
      ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
      // Set PlanDef message size limit to 2G.
      coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
      return proto->ParseFromCodedStream(&coded_stream);
    */
}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn read_proto_from_text_file<M: ::protobuf::Message>(
    filename: *const u8,
    proto:    *mut M) -> bool 
{
    
    todo!();
    /*
        int fd = open(filename, O_RDONLY);
      CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
      FileInputStream* input = new FileInputStream(fd);
      bool success = google::protobuf::TextFormat::Parse(input, proto);
      delete input;
      close(fd);
      return success;
    */
}

#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn write_proto_to_text_file<M: ::protobuf::Message>(
    proto:          &M,
    filename:       *const u8,
    throw_if_error: Option<bool>)  
{
    let throw_if_error: bool = throw_if_error.unwrap_or(true);
    
    todo!();
    /*
        int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      FileOutputStream* output = new FileOutputStream(fd);
      if(!google::protobuf::TextFormat::Print(proto, output)) {
         if (throwIfError) {
           CAFFE_THROW("Cannot write proto to text file: ", filename);
         } else {
           LOG(ERROR) << "Cannot write proto to text file: " << filename;
         }
      }
      delete output;
      close(fd);
    */
}

/**
  | Common interfaces that are supported
  | by both lite and full protobuf.
  |
  */
#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn read_proto_lite_from_binary_file(
    filename: *const u8,
    proto:    *mut MessageLite) -> bool 
{
    
    todo!();
    /*
        #if defined(_MSC_VER) // for MSC compiler binary flag needs to be specified
      int fd = open(filename, O_RDONLY | O_BINARY);
    #else
      int fd = open(filename, O_RDONLY);
    #endif
      CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
      std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
      std::unique_ptr<CodedInputStream> coded_input(
          new CodedInputStream(raw_input.get()));
      // A hack to manually allow using very large protocol buffers.
      coded_input->SetTotalBytesLimit(2147483647, 536870912);
      bool success = proto->ParseFromCodedStream(coded_input.get());
      coded_input.reset();
      raw_input.reset();
      close(fd);
      return success;
    */
}


#[cfg(not(caffe2_use_lite_proto))]
#[inline] pub fn write_proto_lite_to_binary_file(
    proto:    &MessageLite,
    filename: *const u8)  {
    
    todo!();
    /*
        int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
      CAFFE_ENFORCE_NE(
          fd, -1, "File cannot be created: ", filename, " error number: ", errno);
      std::unique_ptr<ZeroCopyOutputStream> raw_output(new FileOutputStream(fd));
      std::unique_ptr<CodedOutputStream> coded_output(
          new CodedOutputStream(raw_output.get()));
      CAFFE_ENFORCE(proto.SerializeToCodedStream(coded_output.get()));
      coded_output.reset();
      raw_output.reset();
      close(fd);
    */
}

/**
  | Helper function to verify that conversion
  | between types won't loose any significant
  | bit.
  |
  */
#[inline] pub fn supports_lossless_conversion<InputType, TargetType>(value: &InputType) -> bool {
    todo!();
    /*
        return static_cast<InputType>(static_cast<TargetType>(value)) == value;
    */
}

impl Eq for TensorProto {}

impl fmt::Display for TensorProto {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            output << n.SerializeAsString();
      return output;
        */
    }
}

impl Eq for QTensorProto {}

impl fmt::Display for QTensorProto {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            output << n.SerializeAsString();
      return output;
        */
    }
}

impl Eq for NetDef {}

impl fmt::Display for NetDef {
    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
        /*
            output << n.SerializeAsString();
      return output;
        */
    }
}

#[macro_export] macro_rules! instantiate_get_single_argument {
    ($T:ident, $fieldname:ident, $enforce_lossless_conversion:ident) => {
        /*
        template <>                                                          
            C10_EXPORT T ArgumentHelper::GetSingleArgument<T>(                   
                const string& name, const T& default_value) const {              
                if (arg_map_.count(name) == 0) {                                   
                    VLOG(1) << "Using default parameter value " << default_value     
                        << " for parameter " << name;                            
                    return default_value;                                            
                }                                                                  
                CAFFE_ENFORCE(                                                     
                    arg_map_.at(name).has_##fieldname(),                           
                    "Argument ",                                                   
                    name,                                                          
                    " does not have the right field: expected field " #fieldname); 
                auto value = arg_map_.at(name).fieldname();                        
                if (enforce_lossless_conversion) {                                 
                    auto supportsConversion =                                        
                        SupportsLosslessConversion<decltype(value), T>(value);       
                    CAFFE_ENFORCE(                                                   
                        supportsConversion,                                          
                        "Value",                                                     
                        value,                                                       
                        " of argument ",                                             
                        name,                                                        
                        "cannot be represented correctly in a target type");         
                }                                                                  
                return static_cast<T>(value);                                      
            }                                                                    
        template <>                                                          
            C10_EXPORT bool ArgumentHelper::HasSingleArgumentOfType<T>(          
                const string& name) const {                                      
                if (arg_map_.count(name) == 0) {                                   
                    return false;                                                    
                }                                                                  
                return arg_map_.at(name).has_##fieldname();                        
            }
        */
    }
}

instantiate_get_single_argument!{f32,      f, false}
instantiate_get_single_argument!{f64,      f, false}
instantiate_get_single_argument!{bool,     i, false}
instantiate_get_single_argument!{i8,       i, true}
instantiate_get_single_argument!{i16,      i, true}
instantiate_get_single_argument!{i32,      i, true}
instantiate_get_single_argument!{i64,      i, true}
instantiate_get_single_argument!{u8,       i, true}
instantiate_get_single_argument!{u16,      i, true}
instantiate_get_single_argument!{usize,    i, true}
instantiate_get_single_argument!{String,   s, false}
instantiate_get_single_argument!{NetDef,   n, false}

#[macro_export] macro_rules! instantiate_get_repeated_argument {
    ($T:ty, $fieldname:ident, $enforce_lossless_conversion:ident) => {
        /*
        template <>                                                          
            C10_EXPORT vector<T> ArgumentHelper::GetRepeatedArgument<T>(         
                const string& name, const std::vector<T>& default_value) const { 
                if (arg_map_.count(name) == 0) {                                   
                    return default_value;                                            
                }                                                                  
                vector<T> values;                                                  
                for (const auto& v : arg_map_.at(name).fieldname()) {              
                    if (enforce_lossless_conversion) {                               
                        auto supportsConversion =                                      
                            SupportsLosslessConversion<decltype(v), T>(v);             
                        CAFFE_ENFORCE(                                                 
                            supportsConversion,                                        
                            "Value",                                                   
                            v,                                                         
                            " of argument ",                                           
                            name,                                                      
                            "cannot be represented correctly in a target type");       
                    }                                                                
                    values.push_back(static_cast<T>(v));                             
                }                                                                  
                return values;                                                     
            }
        */
    }
}

instantiate_get_repeated_argument!{f32,            floats, false}
instantiate_get_repeated_argument!{f64,            floats, false}
instantiate_get_repeated_argument!{bool,           ints, false}
instantiate_get_repeated_argument!{i8,             ints, true}
instantiate_get_repeated_argument!{i16,            ints, true}
instantiate_get_repeated_argument!{i32,            ints, true}
instantiate_get_repeated_argument!{i64,            ints, true}
instantiate_get_repeated_argument!{u8,             ints, true}
instantiate_get_repeated_argument!{u16,            ints, true}
instantiate_get_repeated_argument!{usize,          ints, true}
instantiate_get_repeated_argument!{String,         strings, false}
instantiate_get_repeated_argument!{NetDef,         nets, false}
instantiate_get_repeated_argument!{TensorProto,    tensors, false}
instantiate_get_repeated_argument!{QTensorProto,   qtensors, false}

#[macro_export] macro_rules! caffe2_make_singular_argument {
    ($T:ty, $fieldname:ident) => {
        /*
        template <>                                                            
            C10_EXPORT Argument MakeArgument(const string& name, const T& value) { 
                Argument arg;                                                        
                arg.set_name(name);                                                  
                arg.set_##fieldname(value);                                          
                return arg;                                                          
            }
        */
    }
}

caffe2_make_singular_argument!{bool,   i}
caffe2_make_singular_argument!{f32,    f}
caffe2_make_singular_argument!{i32,    i}
caffe2_make_singular_argument!{i64,    i}
caffe2_make_singular_argument!{String, s}

#[inline] pub fn make_argument_net_def(name: &String, value: &NetDef) -> Argument {
    
    todo!();
    /*
        Argument arg;
      arg.set_name(name);
      *arg.mutable_n() = value;
      return arg;
    */
}

#[inline] pub fn make_argument(name: &String, value: &MessageLite) -> Argument {
    
    todo!();
    /*
        Argument arg;
      arg.set_name(name);
      arg.set_s(value.SerializeAsString());
      return arg;
    */
}


#[macro_export] macro_rules! caffe2_make_repeated_argument {
    ($T:ty, $fieldname:ident) => {
        /*
        template <>                                       
            C10_EXPORT Argument MakeArgument(                 
                const string& name, const vector<T>& value) { 
                Argument arg;                                   
                arg.set_name(name);                             
                for (const auto& v : value) {                   
                    arg.add_##fieldname(v);                       
                }                                               
                return arg;                                     
            }
        */
    }
}

caffe2_make_repeated_argument!{f32,    floats}
caffe2_make_repeated_argument!{i32,    ints}
caffe2_make_repeated_argument!{i64,    ints}
caffe2_make_repeated_argument!{String, strings}

#[inline] pub fn has_output(
    op:     &OperatorDef,
    output: &String) -> bool 
{
    
    todo!();
    /*
        for (const auto& outp : op.output()) {
        if (outp == output) {
          return true;
        }
      }
      return false;
    */
}

#[inline] pub fn has_input(
    op:    &OperatorDef,
    input: &String) -> bool 
{
    
    todo!();
    /*
        for (const auto& inp : op.input()) {
        if (inp == input) {
          return true;
        }
      }
      return false;
    */
}

/**
  | Return the argument index or -1 if it
  | does not exist.
  |
  */
#[inline] pub fn get_argument_index(
    args: &RepeatedPtrField<Argument>,
    name: &String) -> i32 
{
    todo!();
    /*
        int index = 0;
      for (const Argument& arg : args) {
        if (arg.name() == name) {
          return index;
        }
        index++;
      }
      return -1;
    */
}

/**
  | Helper methods to get an argument from
  | OperatorDef or NetDef given argument
  | name. Throws if argument does not exist.
  |
  */
#[inline] pub fn get_argument_from_operator_def<'a>(
    def:  &'a OperatorDef,
    name: &'a String) -> &'a Argument 
{
    todo!();
    /*
        int index = GetArgumentIndex(def.arg(), name);
      if (index != -1) {
        return def.arg(index);
      } else {
        CAFFE_THROW(
            "Argument named ",
            name,
            " does not exist in operator ",
            ProtoDebugString(def));
      }
    */
}

#[inline] pub fn get_argument_from_net_def<'a>(
    def:  &'a NetDef,
    name: &'a String) -> &'a Argument 
{
    todo!();
    /*
        int index = GetArgumentIndex(def.arg(), name);
      if (index != -1) {
        return def.arg(index);
      } else {
        CAFFE_THROW(
            "Argument named ",
            name,
            " does not exist in net ",
            ProtoDebugString(def));
      }
    */
}

/**
  | Helper methods to get an argument from
  | OperatorDef or NetDef given argument
  | name. Returns nullptr if argument does not
  | exist.
  */
#[inline] pub fn get_argument_ptr_from_operator_def(
    def:  &OperatorDef,
    name: &String) -> *const Argument 
{
    todo!();
    /*
        int index = GetArgumentIndex(def.arg(), name);
      if (index != -1) {
        return &def.arg(index);
      } else {
        return nullptr;
      }
    */
}

#[inline] pub fn get_argument_ptr_from_net_def(
    def:  &NetDef,
    name: &String) -> *const Argument 
{
    todo!();
    /*
        int index = GetArgumentIndex(def.arg(), name);
      if (index != -1) {
        return &def.arg(index);
      } else {
        return nullptr;
      }
    */
}

/**
  | Helper methods to query a boolean argument
  | flag from OperatorDef or NetDef given
  | argument name. If argument does not
  | exist, return default value.
  | 
  | Throws if argument exists but the type
  | is not boolean.
  |
  */
#[inline] pub fn get_flag_argument_from_args(
    args:          &RepeatedPtrField<Argument>,
    name:          &String,
    default_value: Option<bool>) -> bool 
{
    let default_value = default_value.unwrap_or(false);
    
    todo!();
    /*
        int index = GetArgumentIndex(args, name);
      if (index != -1) {
        auto arg = args.Get(index);
        CAFFE_ENFORCE(
            arg.has_i(), "Can't parse argument as bool: ", ProtoDebugString(arg));
        return arg.i();
      }
      return default_value;
    */
}

#[inline] pub fn get_flag_argument_from_operator_def(
    def:           &OperatorDef,
    name:          &String,
    default_value: Option<bool>) -> bool 
{
    let default_value = default_value.unwrap_or(false);

    todo!();
    /*
        return GetFlagArgument(def.arg(), name, default_value);
    */
}

#[inline] pub fn get_flag_argument_from_net_def(
    def:           &NetDef,
    name:          &String,
    default_value: Option<bool>) -> bool 
{
    let default_value = default_value.unwrap_or(false);
    
    todo!();
    /*
        return GetFlagArgument(def.arg(), name, default_value);
    */
}

#[inline] pub fn get_mutable_argument_impl<Def>(
    name:              &String,
    create_if_missing: bool,
    def:               *mut Def) -> *mut Argument {
    todo!();
    /*
        for (int i = 0; i < def->arg_size(); ++i) {
        if (def->arg(i).name() == name) {
          return def->mutable_arg(i);
        }
      }
      // If no argument of the right name is found...
      if (create_if_missing) {
        Argument* arg = def->add_arg();
        arg->set_name(name);
        return arg;
      } else {
        return nullptr;
      }
    */
}

#[inline] pub fn get_mutable_argument_from_operator_def(
    name:              &String,
    create_if_missing: bool,
    def:               *mut OperatorDef) -> *mut Argument 
{
    todo!();
    /*
        return GetMutableArgumentImpl(name, create_if_missing, def);
    */
}

#[inline] pub fn get_mutable_argument_from_net_def(
    name:              &String,
    create_if_missing: bool,
    def:               *mut NetDef) -> *mut Argument {
    
    todo!();
    /*
        return GetMutableArgumentImpl(name, create_if_missing, def);
    */
}

/**
  | Given a net, modify the external inputs/outputs
  | if necessary so that the following conditions
  | are met
  | 
  | - No duplicate external inputs
  | 
  | - No duplicate external outputs
  | 
  | - Going through list of ops in order,
  | all op inputs must be outputs from other
  | ops, or registered as external inputs.
  | 
  | - All external outputs must be outputs
  | of some operators.
  |
  */
#[inline] pub fn cleanup_external_inputs_and_outputs(net: *mut NetDef)  {
    
    todo!();
    /*
        std::vector<std::string> oldExternalInputs;
      for (const auto& input : net->external_input()) {
        oldExternalInputs.emplace_back(input);
      }
      std::vector<std::string> oldExternalOutputs;
      for (const auto& output : net->external_output()) {
        oldExternalOutputs.emplace_back(output);
      }

      net->clear_external_input();
      net->clear_external_output();

      std::set<std::string> inputSet;
      for (const auto& input : oldExternalInputs) {
        if (inputSet.count(input)) {
          // Prevent duplicate external inputs.
          continue;
        }
        inputSet.insert(input);
        net->add_external_input(input);
      }

      // Set of blobs that are external inputs or outputs of some operators.
      std::set<std::string> allOutputs(inputSet.begin(), inputSet.end());
      for (const auto& op : net->op()) {
        for (const auto& input : op.input()) {
          if (inputSet.count(input) || allOutputs.count(input)) {
            continue;
          }
          // Add missing external inputs.
          inputSet.insert(input);
          net->add_external_input(input);
        }
        for (const auto& output : op.output()) {
          allOutputs.insert(output);
        }
      }

      std::set<std::string> outputSet;
      for (const auto& output : oldExternalOutputs) {
        if (!allOutputs.count(output)) {
          continue;
        }
        if (outputSet.count(output)) {
          continue;
        }
        outputSet.insert(output);
        net->add_external_output(output);
      }
    */
}
