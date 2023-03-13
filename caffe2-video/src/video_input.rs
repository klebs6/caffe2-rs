crate::ix!();

pub struct VideoInputOp<Context> {
    base: PrefetchOperator<Context>,

    reader:                            *mut DBReader,
    cpu_context:                       CPUContext,
    prefetched_clip_rgb:               Tensor,
    prefetched_clip_of:                Tensor,
    prefetched_label:                  Tensor,
    prefetched_video_id:               Tensor,
    prefetched_start_frame:            Tensor,

    /// {Context::GetDeviceType()};
    prefetched_clip_rgb_on_device:     Tensor,

    /// {Context::GetDeviceType()};
    prefetched_clip_of_on_device:      Tensor,

    /// {Context::GetDeviceType()};
    prefetched_label_on_device:        Tensor,

    /// {Context::GetDeviceType()};
    prefetched_video_id_on_device:     Tensor,

    /// {Context::GetDeviceType()};
    prefetched_start_frame_on_device:  Tensor,
    batch_size:                        i32,
    clip_per_video:                    i32,
    clip_start_positions:              Vec<i32>,
    mean_rgb:                          Vec<f32>,
    inv_std_rgb:                       Vec<f32>,
    mean_of:                           Vec<f32>,
    inv_std_of:                        Vec<f32>,
    channels_rgb:                      i32,
    channels_of:                       i32,
    crop_size:                         i32,
    scale_h:                           i32,
    scale_w:                           i32,
    short_edge:                        i32,
    jitter_scales:                     Vec<i32>,
    length_rgb:                        i32,
    sampling_rate_rgb:                 i32,
    random_sampling_rate:              i32,
    num_of_required_frame:             i32,
    length_of:                         i32,
    sampling_rate_of:                  i32,
    frame_gap_of:                      i32,
    random_mirror:                     bool,
    num_of_class:                      i32,
    use_local_file:                    bool,
    random_crop:                       bool,
    crop_per_clip:                     i32,
    flow_data_type:                    i32,
    flow_alg_type:                     i32,
    decode_type:                       i32,
    video_res_type:                    i32,
    do_flow_aggregation:               bool,
    image_as_input:                    bool,
    get_rgb:                           bool,
    get_optical_flow:                  bool,
    get_video_id:                      bool,
    get_start_frame:                   bool,
    do_multi_label:                    bool,

    /// thread pool for parse + decode
    num_decode_threads:                i32,
    thread_pool:                       Arc<TaskThreadPool>,
}

register_cpu_operator!{VideoInput, VideoInputOp<CPUContext>}

num_inputs!{VideoInput, (0,1)}

num_outputs!{VideoInput, (2,5)}

tensor_inference_function!{VideoInput, /* (
        [](const OperatorDef& def,
           const vector<TensorShape>& /* unused */ /*in*/) {
          ArgumentHelper helper(def);
          int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
          int clip_per_video =
              helper.GetSingleArgument<int>("clip_per_video", 1);
          int crop_size = helper.GetSingleArgument<int>("crop_size", -1);
          int length_rgb = helper.GetSingleArgument<int>("length_rgb", 0);
          int channels_rgb = helper.GetSingleArgument<int>("channels_rgb", 3);
          int length_of = helper.GetSingleArgument<int>("length_of", 0);
          int channels_of = helper.GetSingleArgument<int>("channels_of", 2);

          // get the flags
          bool get_rgb = helper.GetSingleArgument<bool>("get_rgb", true);
          bool get_optical_flow =
              helper.GetSingleArgument<bool>("get_optical_flow", false);
          bool do_multi_label =
              helper.GetSingleArgument<bool>("do_multi_label", false);
          bool get_video_id =
              helper.GetSingleArgument<bool>("get_video_id", false);
          bool get_start_frame =
              helper.GetSingleArgument<bool>("get_start_frame", false);
          // get starting positions if available
          vector<int> clip_start_positions =
              helper.GetRepeatedArgument<int>("clip_start_positions", {});
          // In case clip_start_positions are given, set the clip_per_video arg
          if (clip_start_positions.size() > 0) {
            clip_per_video = clip_start_positions.size();
          }

          int output_size = 1;
          if (get_rgb) {
            output_size++;
          }
          if (get_optical_flow) {
            output_size++;
          }
          if (get_video_id) {
            output_size++;
          }
          if (get_start_frame) {
            output_size++;
          }

          int index = 0;
          vector<TensorShape> out(output_size);
          CHECK_GT(crop_size, 0);
          batch_size *= clip_per_video;
          if (get_rgb) {
            out[index++] = CreateTensorShape(
                vector<int>{
                    batch_size, channels_rgb, length_rgb, crop_size, crop_size},
                TensorProto::FLOAT);
          }
          if (get_optical_flow) {
            out[index++] = CreateTensorShape(
                vector<int>{
                    batch_size, channels_of, length_of, crop_size, crop_size},
                TensorProto::FLOAT);
          }
          if (!do_multi_label) {
            out[index++] = CreateTensorShape(
                vector<int>{1, batch_size}, TensorProto::INT32);
          } else {
            int num_of_class = helper.GetSingleArgument<int>("num_of_class", 0);
            out[index++] = CreateTensorShape(
                vector<int>{batch_size, num_of_class}, TensorProto::INT32);
          }
          if (get_video_id) {
            out[index++] = CreateTensorShape(
                vector<int64_t>{1, batch_size}, TensorProto::INT64);
          }
          if (get_start_frame) {
            out[index] = CreateTensorShape(
                vector<int>{1, batch_size}, TensorProto::INT32);
          }

          return out;
        }) */
}

no_gradient!{VideoInput}

register_cuda_operator!{VideoInput, VideoInputOp<CUDAContext>}

/*
  using OperatorStorage::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
*/

impl<Context> Drop for VideoInputOp<Context> {

    fn drop(&mut self) {
        todo!();
        /* 
        PrefetchOperator<Context>::Finalize();
       */
    }
}

impl<Context> VideoInputOp<Context> {

    #[inline] pub fn check_params_and_print(&mut self)  {
        
        todo!();
        /*
            // check whether the input parameters are valid or not
      CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be positive.");
      CAFFE_ENFORCE_GT(
          clip_per_video_, 0, "Number of clips per video should be positive.");
      CAFFE_ENFORCE_GT(crop_size_, 0, "Must provide the cropping value.");

      if (!image_as_input_) {
        CAFFE_ENFORCE_GT(
            num_of_required_frame_,
            0,
            "Required number of frames must be positive.");
      }

      if (image_as_input_) {
        CAFFE_ENFORCE_EQ(
            video_res_type_,
            VideoResType::USE_WIDTH_HEIGHT,
            "Currently only USE_WIDTH_HEIGHT option is supported with images");
      }

      if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
        CAFFE_ENFORCE_GT(short_edge_, 0, "Must provide the short edge value.");
        CAFFE_ENFORCE_GE(
            short_edge_,
            crop_size_,
            "The short edge must be no smaller than the crop value.");
      } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
        CAFFE_ENFORCE_GT(scale_h_, 0, "Must provide the scale height value.");
        CAFFE_ENFORCE_GT(scale_w_, 0, "Must provide the scale width value.");
        CAFFE_ENFORCE_GE(
            scale_h_,
            crop_size_,
            "The scaled height must be no smaller than the crop value.");
        CAFFE_ENFORCE_GE(
            scale_w_,
            crop_size_,
            "The scaled width must be no smaller than the crop value.");
      }

      if (jitter_scales_.size() > 0) {
        CAFFE_ENFORCE_GE(
            video_res_type_,
            VideoResType::USE_SHORT_EDGE,
            "Scale jittering is used with short_edge scaling only");
      }

      if (get_rgb_) {
        CAFFE_ENFORCE_GT(length_rgb_, 0, "Must provide rgb clip length.");
        CAFFE_ENFORCE_GT(
            sampling_rate_rgb_, 0, "4 frames for mc2; 2 frames for res3d.");
        CAFFE_ENFORCE_EQ(
            channels_rgb_, mean_rgb_.size(), "Number rgb channels is wrong!");
        CAFFE_ENFORCE_EQ(
            channels_rgb_, inv_std_rgb_.size(), "Number rgb channels is wrong!");
      }

      if (get_optical_flow_) {
        CAFFE_ENFORCE_GT(length_of_, 0, "Must provide optical flow clip length.");
        CAFFE_ENFORCE_GT(
            sampling_rate_of_, 0, "4 frames for mc2; 2 frames for res3d.");
        CAFFE_ENFORCE_EQ(
            channels_of_,
            mean_of_.size(),
            "Number of optical flow channels is wrong!");
        CAFFE_ENFORCE_EQ(
            channels_of_,
            inv_std_of_.size(),
            "Number of optical flow channels is wrong!");
      }

      if (clip_per_video_ > 1) {
        CAFFE_ENFORCE_EQ(
            decode_type_,
            DecodeType::DO_UNIFORM_SMP,
            "Only uniformly sampling is supported when sampling multiple clips!");
      }

      if (do_multi_label_) {
        CAFFE_ENFORCE_GT(
            num_of_class_,
            0,
            "Number of classes must be set when using multiple labels.");
      }

      // print out the parameter settings
      LOG(INFO) << "Creating a clip input op with the following setting: ";
      LOG(INFO) << "    Input Type: " << (image_as_input_ ? "Image" : "Video");
      LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
      LOG(INFO) << "    Outputting in batches of " << batch_size_ << " videos;";
      LOG(INFO) << "    Each video has " << clip_per_video_ << " clips;";
      LOG(INFO) << "    Scaling image to " << scale_h_ << "x" << scale_w_;
      LOG(INFO) << "    Cropping video frame to " << crop_size_
                << (random_mirror_ ? " with " : " without ") << "random mirroring;";
      LOG(INFO) << "    Using " << (random_crop_ ? "random" : "center") << " crop";
      LOG(INFO) << "    Using " << crop_per_clip_ << " spatial crop(s)";

      if (get_rgb_) {
        LOG(INFO) << "    Using a clip of " << length_rgb_ << " rgb frames "
                  << "with " << channels_rgb_ << " channels "
                  << "and a sampling rate of 1:" << sampling_rate_rgb_;
        if (random_sampling_rate_) {
          LOG(INFO) << "random sampling with max:" << random_sampling_rate_;
        }
        for (int i = 0; i < channels_rgb_; i++) {
          LOG(INFO) << "    RGB " << i << "-th channel mean: " << mean_rgb_[i]
                    << " std: " << 1.f / inv_std_rgb_[i];
        }
      }

      if (get_optical_flow_) {
        LOG(INFO) << "    Using a clip of " << length_of_ << " optical flow frames "
                  << "with " << channels_of_ << " channels "
                  << "and a sampling rate of 1:" << sampling_rate_of_
                  << " flow_data_type_: " << flow_data_type_
                  << " flow_alg_type_: " << flow_alg_type_;
        for (int i = 0; i < channels_of_; i++) {
          LOG(INFO) << "    Optical flow" << i
                    << "-th channel mean: " << mean_of_[i]
                    << " std: " << 1.f / inv_std_of_[i];
        }
      }

      if (video_res_type_ == VideoResType::ORIGINAL_RES) {
        LOG(INFO) << "    Use original resolution";
      } else if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
        LOG(INFO) << "    Resize and keep aspect ratio";
      } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
        LOG(INFO) << "    Resize and ignore aspect ratio";
      } else {
        LOG(ERROR) << "    Unknown video resolution type";
      }

      if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
        if (jitter_scales_.size() > 0) {
          LOG(INFO) << "Using scale jittering:";
          for (int idx = 0; idx < jitter_scales_.size(); idx++) {
            LOG(INFO) << "scale " << idx << ": " << jitter_scales_[idx];
          }
        } else {
          LOG(INFO) << "No scale jittering is used.";
        }
      }

      if (decode_type_ == DecodeType::DO_TMP_JITTER) {
        LOG(INFO) << "    Do temporal jittering";
      } else if (decode_type_ == DecodeType::USE_START_FRM) {
        LOG(INFO) << "    Use start_frm for decoding";
      } else if (decode_type_ == DecodeType::DO_UNIFORM_SMP) {
        LOG(INFO) << "    Do uniformly sampling";
      } else {
        LOG(ERROR) << "    Unknown video decoding type";
      }
      if (get_start_frame_) {
        CAFFE_ENFORCE_EQ(
            decode_type_,
            DecodeType::USE_START_FRM,
            "Only decoding with starting frame is supported w/ get start_frame!");
        CAFFE_ENFORCE_EQ(
            clip_per_video_, 1, "get start frame support only clip per video = 1");
      }
        */
    }
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : PrefetchOperator<Context>(operator_def, ws),
          reader_(nullptr),
          batch_size_( OperatorStorage::template GetSingleArgument<int>("batch_size", 0)),
          clip_per_video_( OperatorStorage::template GetSingleArgument<int>("clip_per_video", 1)),
          clip_start_positions_(OperatorStorage::template GetRepeatedArgument<int>( "clip_start_positions", {})),
          channels_rgb_( OperatorStorage::template GetSingleArgument<int>("channels_rgb", 3)),
          channels_of_( OperatorStorage::template GetSingleArgument<int>("channels_of", 2)),
          crop_size_(OperatorStorage::template GetSingleArgument<int>("crop_size", 0)),
          scale_h_(OperatorStorage::template GetSingleArgument<int>("scale_h", 0)),
          scale_w_(OperatorStorage::template GetSingleArgument<int>("scale_w", 0)),
          short_edge_( OperatorStorage::template GetSingleArgument<int>("short_edge", 0)),
          jitter_scales_( OperatorStorage::template GetRepeatedArgument<int>("jitter_scales", {})),
          length_rgb_( OperatorStorage::template GetSingleArgument<int>("length_rgb", 0)),
          sampling_rate_rgb_(OperatorStorage::template GetSingleArgument<int>( "sampling_rate_rgb", 1)),
          random_sampling_rate_(OperatorStorage::template GetSingleArgument<int>( "random_sampling_rate", 0)),
          length_of_(OperatorStorage::template GetSingleArgument<int>("length_of", 0)),
          sampling_rate_of_( OperatorStorage::template GetSingleArgument<int>("sampling_rate_of", 1)),
          frame_gap_of_( OperatorStorage::template GetSingleArgument<int>("frame_gap_of", 1)),
          random_mirror_(OperatorStorage::template GetSingleArgument<bool>( "random_mirror", true)),
          num_of_class_( OperatorStorage::template GetSingleArgument<int>("num_of_class", 0)),
          use_local_file_(OperatorStorage::template GetSingleArgument<bool>( "use_local_file", false)),
          random_crop_( OperatorStorage::template GetSingleArgument<bool>("random_crop", true)),
          crop_per_clip_( OperatorStorage::template GetSingleArgument<int>("crop_per_clip", 1)),
          flow_data_type_( OperatorStorage::template GetSingleArgument<int>("flow_data_type", 0)),
          flow_alg_type_( OperatorStorage::template GetSingleArgument<int>("flow_alg_type", 0)),
          decode_type_( OperatorStorage::template GetSingleArgument<int>("decode_type", 0)),
          video_res_type_( OperatorStorage::template GetSingleArgument<int>("video_res_type", 0)),
          do_flow_aggregation_(OperatorStorage::template GetSingleArgument<bool>( "do_flow_aggregation", true)),
          image_as_input_(OperatorStorage::template GetSingleArgument<bool>( "image_as_input", false)),
          get_rgb_(OperatorStorage::template GetSingleArgument<bool>("get_rgb", true)),
          get_optical_flow_(OperatorStorage::template GetSingleArgument<bool>( "get_optical_flow", false)),
          get_video_id_(OperatorStorage::template GetSingleArgument<bool>( "get_video_id", false)),
          get_start_frame_(OperatorStorage::template GetSingleArgument<bool>( "get_start_frame", false)),
          do_multi_label_(OperatorStorage::template GetSingleArgument<bool>( "do_multi_label", false)),
          num_decode_threads_(OperatorStorage::template GetSingleArgument<int>( "num_decode_threads", 4)),
          thread_pool_(std::make_shared<TaskThreadPool>(num_decode_threads_)) 

      try {
        num_of_required_frame_ = 0;

        // mean and std for normalizing different optical flow data type;
        // Example statistics generated from SOA are shown below, and you may
        // want to change them if you are running on a different dataset;

        // 7 channels: (flow_x, flow_y, flow_magitude, gray, Red, Green, Blue)
        const std::vector<float> InputDataMean = {
            0.0046635, 0.0046261, 0.963986, 102.976, 110.201, 100.64, 95.9966};
        const std::vector<float> InputDataStd = {
            0.972347, 0.755146, 1.43588, 55.3691, 58.1489, 56.4701, 55.3324};

        // if we need RGB as an input
        if (get_rgb_ && !image_as_input_) {
          // how many frames we need for RGB
          num_of_required_frame_ = std::max(
              num_of_required_frame_, (length_rgb_ - 1) * sampling_rate_rgb_ + 1);

          if (random_sampling_rate_) {
            num_of_required_frame_ = std::max(
                num_of_required_frame_,
                (length_rgb_ - 1) * random_sampling_rate_ + 1);
          }

          channels_rgb_ = 3;
          for (int i = 4; i < 7; i++) {
            mean_rgb_.push_back(InputDataMean[i]);
            inv_std_rgb_.push_back(1.f / InputDataStd[i]);
          }
        }

        if (image_as_input_) {
          channels_rgb_ = 3;
          length_rgb_ = 1;
          clip_per_video_ = 1;
          get_optical_flow_ = false;
          get_rgb_ = true;
          sampling_rate_rgb_ = 1;
          for (int i = 4; i < 7; i++) {
            mean_rgb_.push_back(InputDataMean[i]);
            inv_std_rgb_.push_back(1.f / InputDataStd[i]);
          }
        }

        // if we need optical flow as an input
        if (get_optical_flow_) {
          // how many frames we need for optical flow
          num_of_required_frame_ = std::max(
              num_of_required_frame_,
              (length_of_ - 1) * sampling_rate_of_ + frame_gap_of_ + 1);

          // set the parameters for different input data types
          switch (flow_data_type_) {
            case FlowDataType::Flow2C:
              channels_of_ = 2;
              for (int i = 0; i < channels_of_; i++) {
                mean_of_.push_back(InputDataMean[i]);
                inv_std_of_.push_back(1.f / InputDataStd[i]);
              }
              break;

            case FlowDataType::Flow3C:
              channels_of_ = 3;
              for (int i = 0; i < channels_of_; i++) {
                mean_of_.push_back(InputDataMean[i]);
                inv_std_of_.push_back(1.f / InputDataStd[i]);
              }
              break;

            // early fusion with gray
            case FlowDataType::FlowWithGray:
              channels_of_ = 3;
              for (int i = 0; i < 2; i++) {
                mean_of_.push_back(InputDataMean[i]);
                inv_std_of_.push_back(1.f / InputDataStd[i]);
              }
              mean_of_.push_back(InputDataMean[3]);
              inv_std_of_.push_back(1.f / InputDataStd[3]);
              break;

            // early fusion with RGB
            case FlowDataType::FlowWithRGB:
              channels_of_ = 5;
              for (int i = 0; i < 2; i++) {
                mean_of_.push_back(InputDataMean[i]);
                inv_std_of_.push_back(1.f / InputDataStd[i]);
              }
              for (int i = 4; i < 7; i++) {
                mean_of_.push_back(InputDataMean[i]);
                inv_std_of_.push_back(1.f / InputDataStd[i]);
              }
              break;

            default:
              LOG(ERROR) << "Unknown optical flow type " << flow_data_type_;
              break;
          }
        }

        CheckParamsAndPrint();
        // Always need a dbreader, even when using local video files
        CAFFE_ENFORCE_GT(
            operator_def.input_size(), 0, "Need to have a DBReader blob input");

        vector<int64_t> data_shape(5);
        vector<int64_t> label_shape(2);

        // In case clip_start_positions are given, set the clip_per_video arg
        if (clip_start_positions_.size() > 0) {
          clip_per_video_ = clip_start_positions_.size();
        }

        // for RGB data
        data_shape[0] = batch_size_ * clip_per_video_ * crop_per_clip_;
        data_shape[1] = channels_rgb_;
        data_shape[2] = length_rgb_;
        data_shape[3] = crop_size_;
        data_shape[4] = crop_size_;
        ReinitializeTensor(
            &prefetched_clip_rgb_, data_shape, at::dtype<float>().device(CPU));

        // for optical flow data
        data_shape[1] = channels_of_;
        data_shape[2] = length_of_;
        ReinitializeTensor(
            &prefetched_clip_of_, data_shape, at::dtype<float>().device(CPU));

        // If do_multi_label is used, output label is a binary vector
        // of length num_of_class indicating which labels present
        if (do_multi_label_) {
          label_shape[0] = batch_size_ * clip_per_video_ * crop_per_clip_;
          label_shape[1] = num_of_class_;
          ReinitializeTensor(
              &prefetched_label_, label_shape, at::dtype<int>().device(CPU));
        } else {
          ReinitializeTensor(
              &prefetched_label_,
              vector<int64_t>(1, batch_size_ * clip_per_video_ * crop_per_clip_),
              at::dtype<int>().device(CPU));
        }

        ReinitializeTensor(
            &prefetched_video_id_,
            vector<int64_t>(1, batch_size_ * clip_per_video_ * crop_per_clip_),
            at::dtype<int>().device(CPU));
        ReinitializeTensor(
            &prefetched_start_frame_,
            vector<int64_t>(1, batch_size_ * clip_per_video_ * crop_per_clip_),
            at::dtype<int>().device(CPU));

      } catch (const std::exception& exc) {
        std::cerr << "While calling VideoInputOp initialization\n";
        std::cerr << exc.what();
      }
        */
    }
    
    #[inline] pub fn get_labels_from_proto(&mut self, label_proto: &TensorProto, label_data: *mut i32)  {
        
        todo!();
        /*
            int num_clips = clip_per_video_ * crop_per_clip_;
      if (!do_multi_label_) {
        for (int i = 0; i < num_clips; i++) {
          label_data[i] = label_proto.int32_data(0);
        }
      } else {
        // For multiple label case, output label is a binary vector
        // where presented concepts are marked 1
        memset(label_data, 0, sizeof(int) * num_of_class_ * num_clips);
        for (int i = 0; i < num_clips; i++) {
          for (int j = 0; j < label_proto.int32_data_size(); j++) {
            CAFFE_ENFORCE_LT(
                label_proto.int32_data(j),
                num_of_class_,
                "Label should be less than the number of classes.");
            label_data[i * num_of_class_ + label_proto.int32_data(j)] = 1;
          }
        }
      }
        */
    }
    
    #[inline] pub fn get_image_and_labels_from_dbvalue(&mut self, 
        value:      &String,
        height:     &mut i32,
        width:      &mut i32,
        buffer_rgb: &mut Vec<*mut u8>,
        label_data: *mut i32) -> bool 
    {
        todo!();
        /*
          TensorProtos protos;
          CAFFE_ENFORCE(protos.ParseFromString(value));
          const TensorProto& image_proto = protos.protos(0);
          const TensorProto& label_proto = protos.protos(1);

          GetLabelsFromProto(label_proto, label_data);

          cv::Mat src;
          if (image_proto.data_type() == TensorProto::STRING) {
            // encoded image string.
            DCHECK_EQ(image_proto.string_data_size(), 1);
            const string& encoded_image_str = image_proto.string_data(0);
            int encoded_size = encoded_image_str.size();
            // We use a cv::Mat to wrap the encoded str so we do not need a copy.
            src = cv::imdecode(
                cv::Mat(
                    1,
                    &encoded_size,
                    CV_8UC1,
                    const_cast<char*>(encoded_image_str.data())),
                cv::IMREAD_COLOR);
            if (src.rows == 0 || src.cols == 0) {
              throw std::runtime_error("Both rows and cols are 0 for image");
            }
          } else if (image_proto.data_type() == TensorProto::BYTE) {
            // raw image content.
            int src_c = (image_proto.dims_size() == 3) ? image_proto.dims(2) : 1;
            CAFFE_ENFORCE(src_c == 3 || src_c == 1);

            src.create(
                image_proto.dims(0),
                image_proto.dims(1),
                (src_c == 3) ? CV_8UC3 : CV_8UC1);
            memcpy(
                src.ptr<uchar>(0),
                image_proto.byte_data().data(),
                image_proto.byte_data().size());
          } else {
            throw std::runtime_error(
                "Unknown image data type: " +
                caffe2::to_string(image_proto.data_type()));
          }
          CAFFE_ENFORCE(src.isContinuous());

          cv::Mat scaled_img;
          cv::resize(
              src, scaled_img, cv::Size(scale_w_, scale_h_), 0, 0, cv::INTER_AREA);

          cv::Mat img;
          if (channels_rgb_ == src.channels()) {
            img = scaled_img;
          } else {
            cv::cvtColor(
                scaled_img, img, (channels_rgb_ == 1) ? cv::COLOR_BGR2GRAY : cv::COLOR_GRAY2BGR);
          }

          cv::Mat rgb_img;

          if (channels_rgb_ == 1) {
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
          } else {
            rgb_img = img;
          }
          CAFFE_ENFORCE(rgb_img.isContinuous());

          unsigned char* data = new unsigned char[scale_h_ * scale_w_ * channels_rgb_];
          memcpy(
              data,
              rgb_img.data,
              scale_h_ * scale_w_ * channels_rgb_ * sizeof(unsigned char));
          buffer_rgb.push_back(data);
          width = scale_w_;
          height = scale_h_;
          return true;
        */
    }
    
    #[inline] pub fn get_clips_and_labels_from_dbvalue(&mut self, 
        value:            &String,
        height:           &mut i32,
        width:            &mut i32,
        buffer_rgb:       &mut Vec<*mut u8>,
        label_data:       *mut i32,
        video_id_data:    *mut i64,
        start_frame_data: *mut i32,
        randgen:          *mut mt19937::MT19937) -> bool {
        
        todo!();
        /*
          TensorProtos protos;
          int curr_proto_idx = 0;
          CAFFE_ENFORCE(protos.ParseFromString(value));
          const TensorProto& video_proto = protos.protos(curr_proto_idx++);
          const TensorProto& label_proto = protos.protos(curr_proto_idx++);

          int start_frm = 0;
          int num_clips = clip_per_video_ * crop_per_clip_;
          // start_frm is only valid when sampling 1 clip per video without
          // temporal jitterring
          if (decode_type_ == DecodeType::USE_START_FRM) {
            CAFFE_ENFORCE_GE(
                protos.protos_size(),
                curr_proto_idx + 1,
                "Start frm proto not provided");
            const TensorProto& start_frm_proto = protos.protos(curr_proto_idx++);
            start_frm = start_frm_proto.int32_data(0);
            if (get_start_frame_) {
              for (int i = 0; i < num_clips; i++) {
                start_frame_data[i] = start_frm;
              }
            }
          }

          if (get_video_id_) {
            CAFFE_ENFORCE_GE(
                protos.protos_size(), curr_proto_idx + 1, "Video Id not provided");
            const TensorProto& video_id_proto = protos.protos(curr_proto_idx);
            for (int i = 0; i < num_clips; i++) {
              video_id_data[i] = video_id_proto.int64_data(0);
            }
          }

          // assign labels
          GetLabelsFromProto(label_proto, label_data);

          if (use_local_file_) {
            CAFFE_ENFORCE_EQ(
                video_proto.data_type(),
                TensorProto::STRING,
                "Database with a file_list is expected to be string data");
          }

          // initializing the decoding params
          Params params;
          params.maximumOutputFrames_ = MAX_DECODING_FRAMES;
          params.video_res_type_ = video_res_type_;
          params.crop_size_ = crop_size_;
          params.short_edge_ = short_edge_;
          params.outputWidth_ = scale_w_;
          params.outputHeight_ = scale_h_;
          params.decode_type_ = decode_type_;
          params.num_of_required_frame_ = num_of_required_frame_;

          if (jitter_scales_.size() > 0) {
            int select_idx =
                std::uniform_int_distribution<>(0, jitter_scales_.size() - 1)(*randgen);
            params.short_edge_ = jitter_scales_[select_idx];
          }

          char* video_buffer = nullptr; // for decoding from buffer
          std::string video_filename; // for decoding from file
          int encoded_size = 0;
          if (video_proto.data_type() == TensorProto::STRING) {
            const string& encoded_video_str = video_proto.string_data(0);
            if (!use_local_file_) {
              encoded_size = encoded_video_str.size();
              video_buffer = const_cast<char*>(encoded_video_str.data());
            } else {
              video_filename = encoded_video_str;
            }
          } else if (video_proto.data_type() == TensorProto::BYTE) {
            if (!use_local_file_) {
              encoded_size = video_proto.byte_data().size();
              video_buffer = const_cast<char*>(video_proto.byte_data().data());
            } else {
              // TODO: does this works?
              video_filename = video_proto.string_data(0);
            }
          } else {
            CAFFE_ENFORCE(false, "Unknown video data type.");
          }

          DecodeMultipleClipsFromVideo(
              video_buffer,
              video_filename,
              encoded_size,
              params,
              start_frm,
              clip_per_video_,
              clip_start_positions_,
              use_local_file_,
              height,
              width,
              buffer_rgb);
          return true;
        */
    }
    
    #[inline] pub fn decode_and_transform(&mut self, 
        value:            &String,
        clip_rgb_data:    *mut f32,
        clip_of_data:     *mut f32,
        label_data:       *mut i32,
        video_id_data:    *mut i64,
        start_frame_data: *mut i32,
        randgen:          *mut mt19937::MT19937,
        mirror_this_clip: *mut statrs::distribution::Bernoulli)  {

        todo!();
        /*
            try {
        std::vector<unsigned char*> buffer_rgb;
        // get the video resolution after decoding
        int height = 0;
        int width = 0;

        if (image_as_input_) {
          CHECK(GetImageAndLabelsFromDBValue(
              value, height, width, buffer_rgb, label_data));
        } else {
          // Decode the video from memory or read from a local file
          CHECK(GetClipsAndLabelsFromDBValue(
              value,
              height,
              width,
              buffer_rgb,
              label_data,
              video_id_data,
              start_frame_data,
              randgen));
        }
        int clip_offset_rgb = channels_rgb_ * length_rgb_ * crop_size_ * crop_size_;
        int clip_offset_of = channels_of_ * length_of_ * crop_size_ * crop_size_;
        for (int i = 0; i < std::min(clip_per_video_, int(buffer_rgb.size()));
             i++) {
          for (int j = 0; j < crop_per_clip_; j++) {
            // get the rectangle for cropping
            int h_off = 0;
            int w_off = 0;
            if (crop_per_clip_ > 1) {
              CAFFE_ENFORCE(
                  random_crop_ == false,
                  "Only using multiple crops w/o random cropping");
            }
            if (random_crop_) {
              // using random crop for training
              h_off =
                  std::uniform_int_distribution<>(0, height - crop_size_)(*randgen);
              w_off =
                  std::uniform_int_distribution<>(0, width - crop_size_)(*randgen);
            } else {
              // using multiple spatial crops
              if (crop_per_clip_ > 1) { // normally 3 crops
                if (height < width) {
                  h_off = (height - crop_size_) / 2;
                  w_off = j * (width - crop_size_) / (crop_per_clip_ - 1);
                } else {
                  h_off = j * (height - crop_size_) / (crop_per_clip_ - 1);
                  w_off = (width - crop_size_) / 2;
                }
                // LOG(INFO) << "crop " << j << "-th " << h_off << " & " << w_off;
              } else { // using center crop for testing
                h_off = (height - crop_size_) / 2;
                w_off = (width - crop_size_) / 2;
              }
            }
            cv::Rect rect(w_off, h_off, crop_size_, crop_size_);

            int this_clip_sampling_rate;
            if (random_sampling_rate_) {
              this_clip_sampling_rate = std::uniform_int_distribution<>(
                  1, random_sampling_rate_)(*randgen);
            }

            // randomly mirror the image or not
            bool mirror_me = random_mirror_ && (*mirror_this_clip)(*randgen);

            if (get_rgb_ && clip_rgb_data) {
              ClipTransformRGB(
                  buffer_rgb[i],
                  crop_size_,
                  length_rgb_,
                  channels_rgb_,
                  (random_sampling_rate_ == 0 ? sampling_rate_rgb_
                                              : this_clip_sampling_rate),
                  height,
                  width,
                  h_off,
                  w_off,
                  mirror_me,
                  mean_rgb_,
                  inv_std_rgb_,
                  clip_rgb_data + ((i * crop_per_clip_ + j) * clip_offset_rgb));
            }

            if (get_optical_flow_ && clip_of_data) {
              ClipTransformOpticalFlow(
                  buffer_rgb[i],
                  crop_size_,
                  length_of_,
                  channels_of_,
                  sampling_rate_of_,
                  height,
                  width,
                  rect,
                  channels_rgb_,
                  mirror_me,
                  flow_alg_type_,
                  flow_data_type_,
                  frame_gap_of_,
                  do_flow_aggregation_,
                  mean_of_,
                  inv_std_of_,
                  clip_of_data + ((i * crop_per_clip_ + j) * clip_offset_of));
            }
          }
        }
        if (buffer_rgb.size() > 0) {
          for (int i = 0; i < buffer_rgb.size(); i++) {
            unsigned char* buff = buffer_rgb[i];
            delete[] buff;
          }
        }
        buffer_rgb.clear();
      } catch (const std::exception& exc) {
        std::cerr << "While calling DecodeAndTransform()\n";
        std::cerr << exc.what();
      }
        */
    }
    
    #[inline] pub fn prefetch(&mut self) -> bool {
        
        todo!();
        /*
            try {
        // We will get the reader pointer from input.
        // If we use local clips, db will store the list
        reader_ = &OperatorStorage::Input<db::DBReader>(0);

        // Call mutable_data() once to allocate the underlying memory.
        prefetched_clip_rgb_.mutable_data<float>();
        prefetched_clip_of_.mutable_data<float>();
        prefetched_label_.mutable_data<int>();
        prefetched_video_id_.mutable_data<int64_t>();
        prefetched_start_frame_.mutable_data<int>();

        // Prefetching handled with a thread pool of "decode_threads" threads.
        std::mt19937 meta_randgen(time(nullptr));
        std::vector<std::mt19937> randgen_per_thread;
        for (int i = 0; i < num_decode_threads_; ++i) {
          randgen_per_thread.emplace_back(meta_randgen());
        }

        std::bernoulli_distribution mirror_this_clip(0.5);
        for (int item_id = 0; item_id < batch_size_; ++item_id) {
          std::mt19937* randgen =
              &randgen_per_thread[item_id % num_decode_threads_];

          int frame_size = crop_size_ * crop_size_;
          // get the clip data pointer for the item_id -th example
          float* clip_rgb_data = prefetched_clip_rgb_.mutable_data<float>() +
              frame_size * length_rgb_ * channels_rgb_ * item_id * clip_per_video_ *
                  crop_per_clip_;

          // get the optical flow data for the current clip
          float* clip_of_data = prefetched_clip_of_.mutable_data<float>() +
              frame_size * length_of_ * channels_of_ * item_id * clip_per_video_ *
                  crop_per_clip_;

          // get the label data pointer for the item_id -th example
          int* label_data = prefetched_label_.mutable_data<int>() +
              (do_multi_label_ ? num_of_class_ : 1) * item_id * clip_per_video_ *
                  crop_per_clip_;

          // get the video id data pointer for the item_id -th example
          int64_t* video_id_data = prefetched_video_id_.mutable_data<int64_t>() +
              item_id * clip_per_video_ * crop_per_clip_;

          int* start_frame_data = prefetched_start_frame_.mutable_data<int>() +
              item_id * clip_per_video_ * crop_per_clip_;

          std::string key, value;
          // read data
          reader_->Read(&key, &value);

          thread_pool_->run(std::bind(
              &VideoInputOp<Context>::DecodeAndTransform,
              this,
              std::string(value),
              clip_rgb_data,
              clip_of_data,
              label_data,
              video_id_data,
              start_frame_data,
              randgen,
              &mirror_this_clip));
        } // for over the batch
        thread_pool_->waitWorkComplete();

        // If the context is not CPUContext, we will need to do a copy in the
        // prefetch function as well.
        if (!std::is_same<Context, CPUContext>::value) {
          if (get_rgb_) {
            prefetched_clip_rgb_on_device_.CopyFrom(
                prefetched_clip_rgb_, &context_);
          }
          if (get_optical_flow_) {
            prefetched_clip_of_on_device_.CopyFrom(prefetched_clip_of_, &context_);
          }
          prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
          if (get_video_id_) {
            prefetched_video_id_on_device_.CopyFrom(
                prefetched_video_id_, &context_);
          }
          if (get_start_frame_) {
            prefetched_start_frame_on_device_.CopyFrom(
                prefetched_start_frame_, &context_);
          }
        }
      } catch (const std::exception& exc) {
        std::cerr << "While calling Prefetch()\n";
        std::cerr << exc.what();
      }
      return true;
        */
    }
    
    #[inline] pub fn copy_prefetched(&mut self) -> bool {
        
        todo!();
        /*
            try {
            int index = 0;
            auto type = Context::GetDeviceType();
            if (get_rgb_) {
              auto* clip_rgb_output = OperatorStorage::Output<Tensor>(index++, type);
              if (std::is_same<Context, CPUContext>::value) {
                clip_rgb_output->CopyFrom(prefetched_clip_rgb_, &context_);
              } else {
                clip_rgb_output->CopyFrom(prefetched_clip_rgb_on_device_, &context_);
              }
            }

            if (get_optical_flow_) {
              auto* clip_of_output = OperatorStorage::Output<Tensor>(index++, type);
              if (std::is_same<Context, CPUContext>::value) {
                clip_of_output->CopyFrom(prefetched_clip_of_, &context_);
              } else {
                clip_of_output->CopyFrom(prefetched_clip_of_on_device_, &context_);
              }
            }

            auto* label_output = OperatorStorage::Output<Tensor>(index++, type);
            if (std::is_same<Context, CPUContext>::value) {
              label_output->CopyFrom(prefetched_label_, &context_);
            } else {
              label_output->CopyFrom(prefetched_label_on_device_, &context_);
            }

            if (get_video_id_) {
              auto* video_id_output = OperatorStorage::Output<Tensor>(index++, type);
              if (std::is_same<Context, CPUContext>::value) {
                video_id_output->CopyFrom(prefetched_video_id_, &context_);
              } else {
                video_id_output->CopyFrom(prefetched_video_id_on_device_, &context_);
              }
            }
            if (get_start_frame_) {
              auto* start_frame_output = OperatorStorage::Output<Tensor>(index, type);
              if (std::is_same<Context, CPUContext>::value) {
                start_frame_output->CopyFrom(prefetched_start_frame_, &context_);
              } else {
                start_frame_output->CopyFrom(
                    prefetched_start_frame_on_device_, &context_);
              }
            }
          } catch (const std::exception& exc) {
            std::cerr << "While calling CopyPrefetched()\n";
            std::cerr << exc.what();
          }

          return true;
        */
    }
}
