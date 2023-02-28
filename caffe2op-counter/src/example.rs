crate::ix!();

/**
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/counter_ops.cc
  |
  */
#[test] fn count_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    createcounter_op = core.CreateOperator(
        "CreateCounter",
        [],
        ["counter"],
        init_count=5
    )

    retrievecount_op = core.CreateOperator(
        "RetrieveCount",
        ["counter"],
        ["count"]
    )

    checkcounterdone_op = core.CreateOperator(
        "CheckCounterDone",
        ["counter"],
        ["done"]
    )

    countup_op = core.CreateOperator(
        "CountUp",
        ["counter"],
        ["previous_count"],
    )

    countdown_op = core.CreateOperator(
        "CountDown",
        ["counter"],
        ["done"],
    )

    resetcounter_op = core.CreateOperator(
        "ResetCounter",
        ["counter"],
        ["previous_count"],
        init_count=3
    )

    // Create counter
    workspace.RunOperatorOnce(createcounter_op)
    print("'counter' pointer:", workspace.FetchBlob("counter"))


    // Retrieve initial counter value
    workspace.RunOperatorOnce(retrievecount_op)
    print("Initial 'count':", workspace.FetchBlob("count"))

    // Check if counter is done
    workspace.RunOperatorOnce(checkcounterdone_op)
    print("Initial 'done' value:", workspace.FetchBlob("done"))

    // Test CountUp operator
    print("\nTesting CountUp operator...")
    for i in range(5):
        workspace.RunOperatorOnce(countup_op)
        print("'previous_count' after CountUp:", workspace.FetchBlob("previous_count"))

    workspace.RunOperatorOnce(retrievecount_op)
    print("'count' value after CountUp test:", workspace.FetchBlob("count"))


    // Test CountDown operator
    print("\nTesting CountDown operator...")
    for i in range(11):
        workspace.RunOperatorOnce(countdown_op)
        workspace.RunOperatorOnce(retrievecount_op)
        print("'count' value after CountDown: {}\t'done' value: {}".format(workspace.FetchBlob("count"), workspace.FetchBlob("done")))

    result:
    'counter' pointer: counter, a C++ native class of type std::__1::unique_ptr<caffe2::Counter<long long>, std::__1::default_delete<caffe2::Counter<long long> > >.
    Initial 'count': 5
    Initial 'done' value: False

    Testing CountUp operator...
    'previous_count' after CountUp: 5
    'previous_count' after CountUp: 6
    'previous_count' after CountUp: 7
    'previous_count' after CountUp: 8
    'previous_count' after CountUp: 9
    'count' value after CountUp test: 10

    Testing CountDown operator...
    'count' value after CountDown: 9        'done' value: False
    'count' value after CountDown: 8        'done' value: False
    'count' value after CountDown: 7        'done' value: False
    'count' value after CountDown: 6        'done' value: False
    'count' value after CountDown: 5        'done' value: False
    'count' value after CountDown: 4        'done' value: False
    'count' value after CountDown: 3        'done' value: False
    'count' value after CountDown: 2        'done' value: False
    'count' value after CountDown: 1        'done' value: False
    'count' value after CountDown: 0        'done' value: False
    'count' value after CountDown: -1        'done' value: True

    */
}

