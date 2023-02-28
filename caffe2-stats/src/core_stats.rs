crate::ix!();

pub struct StatValue {
    v:  Atomic<i64>, // default = 0
}

impl StatValue {
    
    #[inline] pub fn increment(&mut self, inc: i64) -> i64 {
        
        todo!();
        /*
            return v_ += inc;
        */
    }
    
    #[inline] pub fn reset(&mut self, value: Option<i64>) -> i64 {

        let value: i64 = value.unwrap_or(0);

        todo!();
        /*
            return v_.exchange(value);
        */
    }
    
    #[inline] pub fn get(&self) -> i64 {
        
        todo!();
        /*
            return v_.load();
        */
    }
}

pub struct ExportedStatValue {
    key:    String,
    value:  i64,
    ts:     std::time::Instant,
}

/**
  | -----------
  | @brief
  | 
  | Holds names and values of counters exported
  | from a StatRegistry.
  |
  */
pub type ExportedStatList = Vec<ExportedStatValue>;
pub type ExportedStatMap  = HashMap<String, i64>;

#[inline] pub fn to_map(stats: &ExportedStatList) -> ExportedStatMap {
    
    todo!();
    /*
        ExportedStatMap statMap;
  for (const auto& stat : stats) {
    // allow for multiple instances of a key
    statMap[stat.key] += stat.value;
  }
  return statMap;
    */
}

/**
 | @brief Holds a map of atomic counters keyed by
 | name.
 |
 | The StatRegistry singleton, accessed through
 | StatRegistry::get(), holds counters registered
 |  through the macro CAFFE_EXPORTED_STAT. Example
 |  of usage:
 |
 | struct MyCaffeClass {
 |   MyCaffeClass(const std::string& instanceName): stats_(instanceName) {}
 |   void run(int numRuns) {
 |     try {
 |       CAFFE_EVENT(stats_, num_runs, numRuns);
 |       tryRun(numRuns);
 |       CAFFE_EVENT(stats_, num_successes);
 |     } catch (std::exception& e) {
 |       CAFFE_EVENT(stats_, num_failures, 1, "arg_to_usdt", e.what());
 |     }
 |     CAFFE_EVENT(stats_, usdt_only, 1, "arg_to_usdt");
 |   }
 |  private:
 |   struct MyStats {
 |     CAFFE_STAT_CTOR(MyStats);
 |     CAFFE_EXPORTED_STAT(num_runs);
 |     CAFFE_EXPORTED_STAT(num_successes);
 |     CAFFE_EXPORTED_STAT(num_failures);
 |     CAFFE_STAT(usdt_only);
 |   } stats_;
 | };
 |
 | int main() {
 |   MyCaffeClass a("first");
 |   MyCaffeClass b("second");
 |   for (int i = 0; i < 10; ++i) {
 |     a.run(10);
 |     b.run(5);
 |   }
 |   ExportedStatList finalStats;
 |   StatRegistry::get().publish(finalStats);
 | }
 |
 | For every new instance of MyCaffeClass, a new
 | counter is created with the instance name as
 |  prefix. Everytime run() is called, the
 |  corresponding counter will be incremented by
 |  the given value, or 1 if value not provided.
 |
 | Counter values can then be exported into an
 | ExportedStatList. In the example above,
 |  considering "tryRun" never throws, `finalStats`
 |  will be populated as follows:
 |
 |   first/num_runs       100
 |   first/num_successes   10
 |   first/num_failures     0
 |   second/num_runs       50
 |   second/num_successes  10
 |   second/num_failures    0
 |
 | The event usdt_only is not present in
 | ExportedStatList because it is declared as
 | CAFFE_STAT, which does not create a counter.
 |
 | Additionally, for each call to CAFFE_EVENT,
 | a USDT probe is generated.
 |
 | The probe will be set up with the following arguments:
 |   - Probe name: field name (e.g. "num_runs")
 |   - Arg #0: instance name (e.g. "first", "second")
 |   - Arg #1: For CAFFE_EXPORTED_STAT, value of the updated counter
 |             For CAFFE_STAT, -1 since no counter is available
 |   - Args ...: Arguments passed to CAFFE_EVENT, including update value
 |             when provided.
 |
 | It is also possible to create additional
 | StatRegistry instances beyond the
 |  singleton. These instances are not
 |  automatically populated with
 |  CAFFE_EVENT. Instead, they can be populated
 |  from an ExportedStatList structure by calling
 |  StatRegistry::update().
 |
 */
pub struct StatRegistry {
    mutex:  parking_lot::RawMutex,
    stats:  HashMap<String,Box<StatValue>>,
}

impl StatRegistry {

    #[inline] pub fn publish(&mut self, reset: Option<bool>) -> ExportedStatList {

        let reset: bool = reset.unwrap_or(false);

        todo!();
        /*
            ExportedStatList stats;
        publish(stats, reset);
        return stats;
        */
    }
}

///----------------------
pub struct Stat {
    group_name:  String,
    name:        String,
}

impl Stat {

    pub fn new(gn: &String, n: &String) -> Self {
    
        todo!();
        /*
            : groupName(gn), name(n)
        */
    }
    
    #[inline] pub fn increment<Unused>(&mut self, x: Unused) -> i64 {
    
        todo!();
        /*
            return -1;
        */
    }
}

///-------------------------
pub struct ExportedStat {
    base: Stat,
    value: *mut StatValue,
}

impl ExportedStat {
    
    pub fn new(gn: &String, n: &String) -> Self {
    
        todo!();
        /*
            : Stat(gn, n), value_(StatRegistry::get().add(gn + "/" + n))
        */
    }
    
    #[inline] pub fn increment(&mut self, value: Option<i64>) -> i64 {

        let value: i64 = value.unwrap_or(1);

        todo!();
        /*
            return value_->increment(value);
        */
    }
}

///----------------------------
pub struct AvgExportedStat {
    base:  ExportedStat,
    count: ExportedStat,
}

impl AvgExportedStat {

    pub fn new(gn: &String, n: &String) -> Self {
    
        todo!();
        /*
            : ExportedStat(gn, n + "/sum"), count_(gn, n + "/count")
        */
    }
    
    #[inline] pub fn increment(&mut self, value: Option<i64>) -> i64 {

        let value: i64 = value.unwrap_or(1);

        todo!();
        /*
            count_.increment();
        return ExportedStat::increment(value);
        */
    }
}

///-------------
pub struct StdDevExportedStat {
    base: ExportedStat,

    count:        ExportedStat,
    sumsqoffset:  ExportedStat,
    sumoffset:    ExportedStat,

    /// {int64_t::min};
    first:        Atomic<i64>,

    /// {int64_t::min};
    const_min:    i64,
}

impl StdDevExportedStat {

    /**
      | Uses an offset (first_) to remove issue of
      | cancellation
      |
      | Variance is then (sumsqoffset_
      | - (sumoffset_^2) / count_) / (count_ - 1)
      */
    pub fn new(gn: &String, n: &String) -> Self {
    
        todo!();
        /*
            : ExportedStat(gn, n + "/sum"),
            count_(gn, n + "/count"),
            sumsqoffset_(gn, n + "/sumsqoffset"),
            sumoffset_(gn, n + "/sumoffset")
        */
    }
    
    #[inline] pub fn increment(&mut self, value: Option<i64>) -> i64 {

        let value: i64 = value.unwrap_or(1);

        todo!();
        /*
            first_.compare_exchange_strong(const_min_, value);
        int64_t offset_value = first_.load();
        int64_t orig_value = value;
        value -= offset_value;
        count_.increment();
        sumsqoffset_.increment(value * value);
        sumoffset_.increment(value);
        return ExportedStat::increment(orig_value);
        */
    }
}

///------------------------------------
pub struct DetailedExportedStat {
    base:    ExportedStat,
    details: Vec<ExportedStat>,
}

impl DetailedExportedStat {

    pub fn new(gn: &String, n: &String) -> Self {
    
        todo!();
        /*
            : ExportedStat(gn, n)
        */
    }
    
    #[inline] pub fn set_details(&mut self, detail_names: &Vec<String>)  {
        
        todo!();
        /*
            details_.clear();
        for (const auto& detailName : detailNames) {
          details_.emplace_back(groupName, name + "/" + detailName);
        }
        */
    }
    
    #[inline] pub fn increment<T, Unused>(
        &mut self, 
        value:        T,
        detail_index: usize,
        y:            Unused) -> i64 {

        todo!();
        /*
            if (detailIndex < details_.size()) {
          details_[detailIndex].increment(value);
        }
        return ExportedStat::increment(value);
        */
    }
}

///-----------------
pub struct StaticStat {
    base:  Stat,
    value: *mut StatValue,
}

impl StaticStat {
    
    pub fn new(group_name: &String, name: &String) -> Self {
    
        todo!();
        /*
            : Stat(groupName, name),
            value_(StatRegistry::get().add(groupName + "/" + name))
        */
    }
    
    #[inline] pub fn increment(&mut self, value: Option<i64>) -> i64 {
        let value: i64 = value.unwrap_or(1);

        todo!();
        /*
            return value_->reset(value);
        */
    }
}

///--------------------------------------
pub struct _ScopeGuard<T> {
    f:  T,
    start: std::time::Instant,
}

impl<T> _ScopeGuard<T> {
    
    pub fn new(f: T) -> Self {
    
        todo!();
        /*
            : f_(f), start_(std::chrono::high_resolution_clock::now())
        */
    }
    
    #[inline] pub fn scope_guard(&mut self, f: T) -> _ScopeGuard<T> {
    
        todo!();
        /*
            return _ScopeGuard<T>(f);
        */
    }
}

impl<T> Drop for _ScopeGuard<T> {
    fn drop(&mut self) {
        todo!();
        /* 
        using namespace std::chrono;
        auto duration = high_resolution_clock::now() - start_;
        int64_t nanos = duration_cast<nanoseconds>(duration).count();
        f_(nanos);
       */
    }
}

impl<T> Into<bool> for _ScopeGuard<T> {

    /**
      | Using implicit cast to bool so that it
      | can be used in an 'if' condition within
      | 
      | CAFFE_DURATION macro below.
      |
      */
    fn into(self) -> bool {
        true
    }
}

#[macro_export] macro_rules! caffe_stat_ctor {
    ($ClassName:ident) => {
        todo!();
        /*
        
          ClassName(std::string name) : groupName(name) {} 
          std::string groupName
        */
    }
}

#[macro_export] macro_rules! caffe_exported_stat {
    ($name:ident) => {
        todo!();
        /*
        
          ExportedStat name {             
            groupName, #name              
          }
        */
    }
}

#[macro_export] macro_rules! caffe_avg_exported_stat {
    ($name:ident) => {
        todo!();
        /*
        
          AvgExportedStat name {              
            groupName, #name                  
          }
        */
    }
}

#[macro_export] macro_rules! caffe_stddev_exported_stat {
    ($name:ident) => {
        todo!();
        /*
        
          StdDevExportedStat name {              
            groupName, #name                     
          }
        */
    }
}

#[macro_export] macro_rules! caffe_detailed_exported_stat {
    ($name:ident) => {
        todo!();
        /*
        
          DetailedExportedStat name {              
            groupName, #name                       
          }
        */
    }
}


#[macro_export] macro_rules! caffe_stat {
    ($name:ident) => {
        todo!();
        /*
        
          Stat name {            
            groupName, #name     
          }
        */
    }
}

#[macro_export] macro_rules! caffe_static_stat {
    ($name:ident) => {
        todo!();
        /*
        
          StaticStat name {             
            groupName, #name            
          }
        */
    }
}

#[macro_export] macro_rules! caffe_event {
    ($stats:ident, $field:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          {                                                                 
            auto __caffe_event_value_ = stats.field.increment(__VA_ARGS__); 
            CAFFE_SDT(                                                      
                field,                                                      
                stats.field.groupName.c_str(),                              
                __caffe_event_value_,                                       
                ##__VA_ARGS__);                                             
          }
        */
    }
}

#[macro_export] macro_rules! caffe_duration {
    ($stats:ident, $field:ident, $($arg:ident),*) => {
        todo!();
        /*
        
          if (auto g = ::caffe2::detail::ScopeGuard([&](int64_t nanos) { 
                CAFFE_EVENT(stats, field, nanos, ##__VA_ARGS__);         
              }))
        */
    }
}

impl StatRegistry {
    
    /**
      | Add a new counter with given name. If
      | a counter for this name already exists,
      | returns a pointer to it.
      |
      */
    #[inline] pub fn add(&mut self, name: &String) -> *mut StatValue {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      auto it = stats_.find(name);
      if (it != stats_.end()) {
        return it->second.get();
      }
      auto v = std::make_unique<StatValue>();
      auto value = v.get();
      stats_.insert(std::make_pair(name, std::move(v)));
      return value;
        */
    }
    
    /**
      | Populate an ExportedStatList with
      | current counter values.
      | 
      | If `reset` is true, resets all counters
      | to zero. It is guaranteed that no count
      | is lost.
      |
      */
    #[inline] pub fn publish_into_exported_stat_list(&mut self, exported: &mut ExportedStatList, reset: bool)  {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lg(mutex_);
      exported.resize(stats_.size());
      int i = 0;
      for (const auto& kv : stats_) {
        auto& out = exported.at(i++);
        out.key = kv.first;
        out.value = reset ? kv.second->reset() : kv.second->get();
        out.ts = std::chrono::high_resolution_clock::now();
      }
        */
    }
    
    /**
      | Update values of counters contained
      | in the given ExportedStatList to the
      | values provided, creating counters
      | that don't exist.
      |
      */
    #[inline] pub fn update(&mut self, data: &ExportedStatList)  {
        
        todo!();
        /*
            for (const auto& stat : data) {
        add(stat.key)->increment(stat.value);
      }
        */
    }
    
    /**
      | Retrieve the singleton StatRegistry,
      | which gets populated through the CAFFE_EVENT
      | macro.
      |
      */
    #[inline] pub fn get(&mut self) -> &mut StatRegistry {
        
        todo!();
        /*
            static StatRegistry r;
      return r;
        */
    }
}
