/*
| #[test]
| fn test_something_interesting() {
|     run_test(|| {
|         let true_or_false = do_the_test();
|
|         assert!(true_or_false);
|     })
| }
|
| fn run_test<T>(test: T) -> ()
|     where T: FnOnce() -> () + panic::UnwindSafe
| {
|     setup();
|
|     let result = panic::catch_unwind(|| {
|         test()
|     });
|
|     teardown();
|
|     assert!(result.is_ok())
| }
*/

