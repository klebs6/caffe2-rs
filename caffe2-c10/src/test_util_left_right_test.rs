crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/LeftRight_test.cpp]

#[test] fn left_right_test_given_int_when_writing_and_reading_then_changes_are_present() {
    todo!();
    /*
    
      LeftRight<int> obj;

      obj.write([](int& obj) { obj = 5; });
      int read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(5, read);

      // check changes are also present in background copy
      obj.write([](int&) {}); // this switches to the background copy
      read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(5, read);

    */
}

#[test] fn left_right_test_given_vector_when_writing_and_reading_then_changes_are_present() {
    todo!();
    /*
    
      LeftRight<vector<int>> obj;

      obj.write([](vector<int>& obj) { obj.push_back(5); });
      vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
      EXPECT_EQ((vector<int>{5}), read);

      obj.write([](vector<int>& obj) { obj.push_back(6); });
      read = obj.read([](const vector<int>& obj) { return obj; });
      EXPECT_EQ((vector<int>{5, 6}), read);

    */
}

#[test] fn left_right_test_given_vector_when_writing_returns_value_then_is_returned() {
    todo!();
    /*
    
      LeftRight<vector<int>> obj;

      auto a = obj.write([](vector<int>&) -> int { return 5; });
      static_assert(is_same<int, decltype(a)>::value, "");
      EXPECT_EQ(5, a);

    */
}

#[test] fn left_right_test_reads_can_be_concurrent() {
    todo!();
    /*
    
      LeftRight<int> obj;
      atomic<int> num_running_readers{0};

      thread reader1([&]() {
        obj.read([&](const int&) {
          ++num_running_readers;
          while (num_running_readers.load() < 2) {
          }
        });
      });

      thread reader2([&]() {
        obj.read([&](const int&) {
          ++num_running_readers;
          while (num_running_readers.load() < 2) {
          }
        });
      });

      // the threads only finish after both entered the read function.
      // if LeftRight didn't allow concurrency, this would cause a deadlock.
      reader1.join();
      reader2.join();

    */
}

#[test] fn left_right_test_writes_can_be_concurrent_with_reads_read_then_write() {
    todo!();
    /*
    
      LeftRight<int> obj;
      atomic<bool> reader_running{false};
      atomic<bool> writer_running{false};

      thread reader([&]() {
        obj.read([&](const int&) {
          reader_running = true;
          while (!writer_running.load()) {
          }
        });
      });

      thread writer([&]() {
        // run read first, write second
        while (!reader_running.load()) {
        }

        obj.write([&](int&) { writer_running = true; });
      });

      // the threads only finish after both entered the read function.
      // if LeftRight didn't allow concurrency, this would cause a deadlock.
      reader.join();
      writer.join();

    */
}

#[test] fn left_right_test_writes_can_be_concurrent_with_reads_write_then_read() {
    todo!();
    /*
    
      LeftRight<int> obj;
      atomic<bool> writer_running{false};
      atomic<bool> reader_running{false};

      thread writer([&]() {
        obj.read([&](const int&) {
          writer_running = true;
          while (!reader_running.load()) {
          }
        });
      });

      thread reader([&]() {
        // run write first, read second
        while (!writer_running.load()) {
        }

        obj.read([&](const int&) { reader_running = true; });
      });

      // the threads only finish after both entered the read function.
      // if LeftRight didn't allow concurrency, this would cause a deadlock.
      writer.join();
      reader.join();

    */
}

#[test] fn left_right_test_writes_cannot_be_concurrent_with() {
    todo!();
    /*
    
      LeftRight<int> obj;
      atomic<bool> first_writer_started{false};
      atomic<bool> first_writer_finished{false};

      thread writer1([&]() {
        obj.write([&](int&) {
          first_writer_started = true;
          this_thread::sleep_for(chrono::milliseconds(50));
          first_writer_finished = true;
        });
      });

      thread writer2([&]() {
        // make sure the other writer runs first
        while (!first_writer_started.load()) {
        }

        obj.write([&](int&) {
          // expect the other writer finished before this one starts
          EXPECT_TRUE(first_writer_finished.load());
        });
      });

      writer1.join();
      writer2.join();

    */
}

#[derive(Debug,Error)]
pub enum MyException { 
    Default,
}

#[test] fn left_right_test_when_read_throws_exception_then_through() {
    todo!();
    /*
    
      LeftRight<int> obj;

      EXPECT_THROW(obj.read([](const int&) { throw MyException(); }), MyException);

    */
}

#[test] fn left_right_test_when_write_throws_exception_then_through() {
    todo!();
    /*
    
      LeftRight<int> obj;

      EXPECT_THROW(obj.write([](int&) { throw MyException(); }), MyException);

    */
}

#[test] fn left_right_test_given_int_when_write_throws_exception_on_first_call_then_resets_to_old_state() {
    todo!();
    /*
    
      LeftRight<int> obj;

      obj.write([](int& obj) { obj = 5; });

      EXPECT_THROW(
          obj.write([](int& obj) {
            obj = 6;
            throw MyException();
          }),
          MyException);

      // check reading it returns old value
      int read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(5, read);

      // check changes are also present in background copy
      obj.write([](int&) {}); // this switches to the background copy
      read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(5, read);

    */
}

/**
  | note: each write is executed twice, on the
  | foreground and background copy.
  |
  | We need to test a thrown exception in either
  | call is handled correctly.
  |
  */
#[test] fn left_right_test_given_int_when_write_throws_exception_on_second_call_then_keeps_new_state() {
    todo!();
    /*
    
      LeftRight<int> obj;

      obj.write([](int& obj) { obj = 5; });
      bool write_called = false;

      EXPECT_THROW(
          obj.write([&](int& obj) {
            obj = 6;
            if (write_called) {
              // this is the second time the write callback is executed
              throw MyException();
            } else {
              write_called = true;
            }
          }),
          MyException);

      // check reading it returns new value
      int read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(6, read);

      // check changes are also present in background copy
      obj.write([](int&) {}); // this switches to the background copy
      read = obj.read([](const int& obj) { return obj; });
      EXPECT_EQ(6, read);

    */
}

#[test] fn left_right_test_given_vector_when_write_throws_exception_then_resets_to_old_state() {
    todo!();
    /*
    
      LeftRight<vector<int>> obj;

      obj.write([](vector<int>& obj) { obj.push_back(5); });

      EXPECT_THROW(
          obj.write([](vector<int>& obj) {
            obj.push_back(6);
            throw MyException();
          }),
          MyException);

      // check reading it returns old value
      vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
      EXPECT_EQ((vector<int>{5}), read);

      // check changes are also present in background copy
      obj.write([](vector<int>&) {}); // this switches to the background copy
      read = obj.read([](const vector<int>& obj) { return obj; });
      EXPECT_EQ((vector<int>{5}), read);

    */
}
