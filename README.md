# Caffe2-rs

Caffe2-rs is a Rust crate that provides an
operator network library for the Rust
ecosystem. The library is based on the flexible
and robust Caffe2 C++ library and aims to provide
high performance, high modularity, and ease of
integration into all future systems written in
Rust.

The goal of this project is to finish translating
the implementations of each operator and
supporting crates. Although much of the code is
still in need of polish, the interfaces are
roughly in place. The translation task is now
parallelized. To ensure functionality, the unit
tests from the C++ codebase have been included.

In the near future, this library could serve as
a reference model for PyTorch, as well as a tool
for the Rust ecosystem. The project owes its DNA
to the original authors of the Caffe2 C++ library,
without whom this work would not exist.

## Contributing

Contributions to this project are welcome and
encouraged. If you're interested in contributing,
please take a look at the [contribution guidelines](CONTRIBUTING.md).

New as of 3/1/2023 -- discord server `caffe2-rs`
launched!

https://discord.gg/CdHsMHJGX8

## License

Caffe2-rs has a BSD-style license, as found in the
[LICENSE](LICENSE) file.

## README Disclaimer
The README files for each crate are being
generated via a varying length, medium depth
conversation with chatgpt.

I'm not sure what everybody else thinks about
this, but it seems pretty awesome to me.

In certain circumstances I have been able to drill
down into some depth to find out what the bot
knows about the different algorithms.

The only thing to watch out for is this: the
README files are only *probabilistically correct*

I repeat: you can't rely on them totally. 

They serve more as guidelines, than hard and fast
rules.

Sometimes, I left information from these
conversations in the README even though I *knew*
it wasn't entirely accurate. (although it may be
accurate one day)

For example, sometimes, the bot generated example
usage code based on its knowledge of just a few
symbols from the crate.  There is probably no way
this generated code works out of the box.   As of
March 6, 2023, I haven't tried all of it.

To me, at least, this information is still useful. 

This is because it represents the bot's best guess
on how this crate might be used.  

It is useful to me to know how the bot might try
and want to use the crate, even if the APIs it is
asking for dont actually exist yet (for example).  

In another example, sometimes I ask the bot to do
some numerical simulations to calculate
performance behavior.  Please don't blindly trust
these numbers.  I'm not totally sure they're
correct even though they look like a pretty good
start.

In other words, please take the information in the
README files only for what it actually is, and
what it actually represents.  

Please consider them a guideline and not
a contract.

I provide this information in case it is
interesting, and so you yourself don't have to
spend the sleep wake cycles asking the bot if you
don't want to.

Nothin but love,

-klebs

BTW if you find glaring errors in these files, plz
feel free to correct them.  Correcting the
possible mistakes is totally welcome.


## Acknowledgments

The author of this project would like to thank the
original authors of the Caffe2 C++ library for
their work, as well as the Rust community for
their support.
