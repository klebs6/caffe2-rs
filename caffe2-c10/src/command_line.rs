crate::ix!();

#[derive(Debug, StructOpt)]
#[structopt(
    name  = "c10", 
    about = "Command line options for customizing the behavior of the c10 library")
]
pub struct C10CommandLineOpts {

    /// Activate debug mode
    #[structopt(short, long)]
    debug: bool,

    /// Equivalent to glog verbose
    ///
    #[structopt(short = "v", long = "verbose")]
    verbose: bool,

    /// If set, print out detailed memory usage
    ///
    #[structopt(long)]
    caffe2_report_cpu_memory_usage: bool,

    /// If set, do memory zerofilling when
    /// allocating on CPU"
    ///
    #[structopt(long)]
    caffe2_cpu_allocator_do_zero_fill: bool,

    /// If set, fill memory with deterministic
    /// junk when allocating on CPU
    ///
    #[structopt(long)]
    caffe2_cpu_allocator_do_junk_fill: bool,

    /// Use NUMA whenever possible.
    #[structopt(long)]
    caffe2_cpu_numa_enabled: bool,

    /// If set true, when CAFFE_ENFORCE is not
    /// met, abort instead of throwing an
    /// exception.
    ///
    #[structopt(long)]
    caffe2_use_fatal_for_enforce: bool,

    /// Equivalent to glog minloglevel
    ///
    #[structopt(long, default_value = "0")]
    minloglevel: usize,

    /// Equivalent to glog logtostderr
    ///
    #[structopt(long)]
    logtostderr: bool,

    /// The minimum log level that caffe2 will
    /// output.
    ///
    #[structopt(long, default_value = "WARNING")]
    caffe2_log_level: usize,

    /// A global boolean variable to control
    /// whether we free memory when a Tensor is
    /// shrunk to a smaller size. As a result,
    /// a Tensor is always going to keep the
    /// memory allocated for its maximum capacity
    /// reshaped to so far.
    /// 
    /// This parameter is respected "upper-case"
    /// methods which call Resize() (e.g.,
    /// CopyFrom, ResizeLike); it is NOT respected
    /// by Tensor::resize_ or ShrinkTo, both of
    /// which guarantee to never to free memory.
    ///
    /// If set, keeps memory when a tensor is
    /// shrinking its size.
    ///
    #[structopt(long)]
    no_caffe2_keep_on_shrink: bool,

    /// The maximum memory in bytes to keep on
    /// shrink, if the difference between tensor
    /// sizes is bigger than this then tensor will
    /// be reset
    ///
    /// Since we can have high variance in blob
    /// memory allocated across different inputs
    /// in the same run, we will shrink the blob
    /// only if the memory gain is larger than
    /// this flag in bytes.  This only applies to
    /// functions which respect
    /// caffe2_keep_on_shrink.
    ///
    #[structopt(long, default_value = "i64::MAX")]
    caffe2_max_keep_on_shrink_memory: i64,
}
