#!/usr/bin/env raku
use v6;

#this one is for 
#use crate::X;
#use crate::Y;
#use crate::Z;
#
#all in a row just like that
#
grammar UseStatements {
    rule TOP                { <use-statement>+ }
    rule use-statement      { 'use crate::' <maybe-scoped-ident> ';' }
    rule ident              { <[A..Z a..z _ 0..9]>+ }
    rule maybe-scoped-ident { [<ident> '::']* <ident> }
}

#this one creates a UseBlock from UseStatements
our sub condense-use($input) {
    UseStatements.parse($input, actions => class {
        method TOP($/) {
            make qq:to/END/;
            use crate::\{
            { $/<use-statement>>>.made.join("\n").chomp.trim.indent(4) }
            \};
            END
        }
        method use-statement($/) {
            make "{$/<maybe-scoped-ident>.Str},"
        }
    }).made
}

#this one is for 
#use crate::{
#    X,
#    Y,
#    Z,
#};
#
#all together in one block
#
role UseBlock {
    rule use-block {
        'use crate::{'
        <use-list>
        '};'
    }
    rule use-list {
        <ident>+ %% ','
    }
    rule ident {
        <[A..Z a..z _]>+
    }
}

role UseBlockActions {
    method use-block($/) {
        make $/<use-list>.made
    }
    method use-list($/) {
        make $/<ident>>>.made
    }
    method ident($/) {
        make $/.Str
    }
}

our sub parse-use-sheet($input, :$just-print = False) {

    #use Grammar::Tracer;

    grammar UseSheetGrammar does UseBlock {
        rule TOP {
            <use-sheet-item>+
        }
        rule use-sheet-item {
            <use-sheet-delim>
            <filename>
            <use-block>?
        }
        rule filename {
            \N+
        }
        rule use-sheet-delim {
            '-'+
        }
    }

    class Actions does UseBlockActions {
        method TOP($/) {
            make $/<use-sheet-item>>>.made
        }
        method use-sheet-item($/) {
            if $/<use-block>:exists {
                make {
                    use-block => $/<use-block>.made,
                    file  => $/<filename>.Str.chomp
                }
            }
        }
    }

    my @items = UseSheetGrammar.parse($input, actions => Actions.new).made;

    for @items {
        if $_ {
            my $file = $_<file>;
            my @use-block = $_<use-block>.List;

            if $just-print {
                say "---";
                say $file;
                say @use-block;
            } else {
                update-uses($file, @use-block);
            }
        }
    }
}

our sub update-uses($file, @use-block) {

    my $contents = $file.IO.slurp;

    grammar HasUseBlock does UseBlock {
        rule TOP {
            [.*? <use-block>]*
        }
    }

    class HasUseBlockActions does UseBlockActions {
        method TOP($/) {
            make $/<use-block>>>.made
        }
    }

    my $parsed = HasUseBlock.subparse($contents,actions => HasUseBlockActions.new);
    my @their-use-block = $parsed.made;

    #say $parsed;
    my $total = SetHash.new;

    for (|@use-block, @their-use-block.List) {
        for $_.List {
            for $_.List {
                $total.set($_.chomp);
            }
        }
    }

    #say "-----------------------------";
    #say $file;
    #say $total;
    #say "";
    my $new-contents = chop-out-parsed-use-blocks($file, $parsed);
    $new-contents = write-use-block($new-contents, $total);

    spurt $file, $new-contents;
}

our sub chop-out-parsed-use-blocks($file, $parsed) {

    my $contents = $file.IO.slurp;

    for $parsed<use-block> -> $use-block {
        my $from = $use-block.from;
        my $to = $use-block.to;
        my $len = $to.Int - $from.Int;
        $contents.substr-rw($from, $len - 1) = " " x $len;
    }

    $contents
}

our sub use-block-from-set(SetHash $set) {
    qq:to/END/.chomp;
    use crate::\{
    {$set.keys.join(",\n").indent(4)}
    \};
    END
}

our sub write-use-block($contents is rw, SetHash $set) {
    my $idx = $contents.index('ix!();') + 7;
    my $use-block = use-block-from-set($set);
    $contents.substr-rw($idx, 1) = "\n$use-block\n\n";
    $contents

}


multi sub MAIN {
    say condense-use($*IN.slurp);
}

multi sub MAIN(:$sheet) {
    parse-use-sheet($*IN.slurp, just-print => False);
}

