crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/test/vitals.cpp]

#[test] fn vitals_basic() {
    todo!();
    /*
    
      stringstream buffer;

      streambuf* sbuf = cout.rdbuf();
      cout.rdbuf(buffer.rdbuf());
      {
    #ifdef _WIN32
        _putenv("TORCH_VITAL=1");
    #else
        setenv("TORCH_VITAL", "1", 1);
    #endif
        TORCH_VITAL_DEFINE(Testing);
        TORCH_VITAL(Testing, Attribute0) << 1;
        TORCH_VITAL(Testing, Attribute1) << "1";
        TORCH_VITAL(Testing, Attribute2) << 1.0f;
        TORCH_VITAL(Testing, Attribute3) << 1.0;
        auto t = ones({1, 1});
        TORCH_VITAL(Testing, Attribute4) << t;
      }
      cout.rdbuf(sbuf);

      auto s = buffer.str();
      ASSERT_TRUE(s.find("Testing.Attribute0\t\t 1") != string::npos);
      ASSERT_TRUE(s.find("Testing.Attribute1\t\t 1") != string::npos);
      ASSERT_TRUE(s.find("Testing.Attribute2\t\t 1") != string::npos);
      ASSERT_TRUE(s.find("Testing.Attribute3\t\t 1") != string::npos);
      ASSERT_TRUE(s.find("Testing.Attribute4\t\t  1") != string::npos);

    */
}

#[test] fn vitals_multi_string() {
    todo!();
    /*
    
      stringstream buffer;

      streambuf* sbuf = cout.rdbuf();
      cout.rdbuf(buffer.rdbuf());
      {
    #ifdef _WIN32
        _putenv("TORCH_VITAL=1");
    #else
        setenv("TORCH_VITAL", "1", 1);
    #endif
        TORCH_VITAL_DEFINE(Testing);
        TORCH_VITAL(Testing, Attribute0) << 1 << " of " << 2;
        TORCH_VITAL(Testing, Attribute1) << 1;
        TORCH_VITAL(Testing, Attribute1) << " of ";
        TORCH_VITAL(Testing, Attribute1) << 2;
      }
      cout.rdbuf(sbuf);

      auto s = buffer.str();
      ASSERT_TRUE(s.find("Testing.Attribute0\t\t 1 of 2") != string::npos);
      ASSERT_TRUE(s.find("Testing.Attribute1\t\t 1 of 2") != string::npos);

    */
}

#[test] fn vitals_on_and_off() {
    todo!();
    /*
    
      for (auto i = 0; i < 2; ++i) {
        stringstream buffer;

        streambuf* sbuf = cout.rdbuf();
        cout.rdbuf(buffer.rdbuf());
        {
    #ifdef _WIN32
          if (i) {
            _putenv("TORCH_VITAL=1");
          } else {
            _putenv("TORCH_VITAL=0");
          }
    #else
          setenv("TORCH_VITAL", i ? "1" : "", 1);
    #endif
          TORCH_VITAL_DEFINE(Testing);
          TORCH_VITAL(Testing, Attribute0) << 1;
        }
        cout.rdbuf(sbuf);

        auto s = buffer.str();
        auto f = s.find("Testing.Attribute0\t\t 1");
        if (i) {
          ASSERT_TRUE(f != string::npos);
        } else {
          ASSERT_TRUE(f == string::npos);
        }
      }

    */
}

#[test] fn vitals_api() {
    todo!();
    /*
    
      stringstream buffer;
      bool rvalue;
      streambuf* sbuf = cout.rdbuf();
      cout.rdbuf(buffer.rdbuf());
      {
    #ifdef _WIN32
        _putenv("TORCH_VITAL=1");
    #else
        setenv("TORCH_VITAL", "1", 1);
    #endif
        APIVitals api_vitals;
        rvalue = api_vitals.setVital("TestingSetVital", "TestAttr", "TestValue");
      }
      cout.rdbuf(sbuf);

      auto s = buffer.str();
      ASSERT_TRUE(rvalue);
      ASSERT_TRUE(s.find("TestingSetVital.TestAttr\t\t TestValue") != string::npos);

    */
}
