crate::ix!();

#[test] fn lt_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LT",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False False  True False False  True]

    */
}

#[test] fn le_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True False  True  True  True  True]

    */
}

#[test] fn gt_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GT",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False  True False False False False]

    */
}

#[test] fn ge_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "GE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True  True False  True  True False]
    
    */
}

#[test] fn eq_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "EQ",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [ True False False  True  True False]
    */
}

#[test] fn ne_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "NE",
        ["A",  "B"],
        ["C"],
    )

    workspace.FeedBlob("A", np.array([1, 5, 2, 9, 12, 3]))
    workspace.FeedBlob("B", np.array([1, 3, 4, 9, 12, 8]))
    print("A:", workspace.FetchBlob("A"))
    print("B:", workspace.FetchBlob("B"))
    workspace.RunOperatorOnce(op)
    print("C:", workspace.FetchBlob("C"))

    A: [ 1  5  2  9 12  3]
    B: [ 1  3  4  9 12  8]
    C: [False  True  True False False  True]
    */
}

