This library is based on the flexible and robust
Caffe2 C++ project and aims to provide high
performance, high modularity, and ease of
integration into future rust systems.

As of July 1, 2023, the codebase is under
construction: it is ready for interface
experimentation and development work, but not yet
for production.

If you are interested in this project and would
like to contribute, your contribution is both
welcome and appreciated!

The current bottleneck is in the translation of
c++ statements into rust. It is possible to do
this manually. However, it is better to do it
automatically. Some work is being done in this
domain, but there are several challenges to be
overcome.

For more information, see `chomper`:
https://github.com/klebs6/chomper

## Note from original translator (klebs6)

Greetings! I have a few preliminary work-items to
attend for a few months. Once these are done,
I plan to rewrite the `chomper`. 

This will include integrations with the
rust-ecosystem compiler tools. 

By rewriting `chomper` in rust, we will be able to
enable name resolution and type inference during
transpilation (not to mention the performance
wins).

This productionizing of the transpiler ought to
provide a *massive* speedup for the transpilation
work. My hope is that it will allow caffe2-rs
development to get moving.

If anybody reading this would like to contribute
to this c++ to rust transpiler development track,
that is great! I would love to hear from you.
Please feel free to reach out. 

I am sure the rust ecosystem will be grateful,
too. C++ to Rust transpilation seems like a major
bottleneck in a lot of places. 

## Acknowledgments

This project owes its DNA to the original authors
of the Caffe2 C++ library, without whom, this work
would not exist.

The author of this project would like to thank the
original authors of the Caffe2 C++ library for
their work, as well as the Rust community for
their support.
