# DAS_cut

`DAS_cut` is a tool to cut down a DAS data file to a subset of its channels
and/or samples.  Since raw DAS data files tend to be very large, it's not
well suited for either sharing or testing.

The program is run as follows:

    python DAS_cut.py [OPTIONS] <input_file> [CUT-SPEC]

Options are as follows:

* `-v` (`--verbose`) enables verbose output

There are two ways to use the program, based on how many output files to
generate.  Both approaches require specifying what portions of the input
file should be included in output files.

## Cut Specifications

A _cut specification_ (or "cut-spec") includes these components:

    <output_file> [channels <channel_filter>] [times <time_filter>]

The `output_file` is the filename of the output file to generate.

If a subset of the input file's channels are to be included then the spec
can include the `channels` keyword and then a channel-filter.

Similarly, to filter down the samples, specify the `times` keyword and then
a filter of what sample-times to include in the output file.

To simplify the program's implementation, all cut specifications are parsed
very simply, following the command shell's parsing rules.  Therefore, if the
`channel_filter` or `time_filter` includes spaces or other special characters
then it needs to be enclosed in single-quotes or double-quotes.

## Channel Filters

A channel filter includes one or more channels or ranges of channels to
include in the output, separated by commas.  The first channel is channel 0.
Here are some examples:

* `channels 100-199` includes channels 100-199 (the second 100 channels)
  in the output file.
* `channels "5, 20, 50-80"` includes 33 channels in the output file.  The
  filter needs to be quoted since it includes spaces.  Alternately it can be
  written `channels 5,20,50-80`.

**Note that this is merely a filtering specification, and cannot be used to
duplicate or reorder channels from the input file.**  For example:

* `channels 150-199,50-99` includes 100 channels in the output, but in the
  same order they appear in the input file
* `channels 50-149,100-199` includes 150 channels in the output, in the same
  order they appear in the input file

## Time Filters

A time filter includes one or more time intervals to include in the output,
separated by commas.  Time intervals are specified by
`<datetime1>~<datetime2>` pairs, with date/time values specified in whatever
formats are supported by the `dateutil.parser.parse()` API function.  Note
that the date/time values are separated by a tilde `"~"` since other
characters like hyphens `"-"` or slashes `"/"` may be used in date/time
values.  Here are some examples:

* `times 20231004T175400Z~20231004T175405Z` includes all samples starting at
  4 Oct 2023, 5:54:00PM UTC (inclusive) up to 4 Oct 2023, 5:54:05PM UTC
  (exclusive) in the output file.
* `times 20231004T175400Z~20231004T175405Z,20231004T175405.5Z~20231004T175408Z`
  includes all samples starting at 4 Oct 2023, 5:54:00PM UTC (inclusive) up
  to 4 Oct 2023, 5:54:05PM UTC (exclusive), and also all samples starting at
  4 Oct 2023, 5:54:05.5PM UTC (inclusive) up to 4 Oct 2023, 5:54:08PM UTC
  (exclusive), in the output file.

**As before, this is merely a filtering specification, and cannot be used to
reorder samples from the input file.**

Also, as before, if a time-filter includes spaces or other special characters
then the filter should be enclosed in quotes.  For example, the second filter
above could be written as follows, including the quotes:

* `times "20231004T175400Z ~ 20231004T175405Z, 20231004T175405.5Z ~ 20231004T175408Z"`

## Command-Line Invocation

The `DAS_cut` program can be invoked from the command-line and a single
cut-spec given on the command line, like this:

    # The $ is the command-prompt
    $ python DAS_cut.py input.h5 output.h5 channels 3000-3999 \
        times "20231004T175400Z ~ 20231004T175405Z"

This will generate a file `output.h5` with only 1000 channels and 5 seconds
of samples from the input file.

Alternately, the program can be invoked with no cut-spec on the command-line,
and instead, one or more cut-specs can be given on `stdin`, like this:

    # The $ is the command-prompt
    $ python DAS_cut.py input.h5
    output1.h5 channels 3000-3999 times "20231004T175400Z ~ 20231004T175405Z"
    output2.h5 channels 3000-3999 times "20231004T175405Z ~ 20231004T175410Z"
    output3.h5 channels 3000-3999 times "20231004T175410Z ~ 20231004T175415Z"
    # Press Ctrl-D to signal EOF

These cut-specs can also be put in a text file and fed into the program's
`stdin` via redirection.

As before, these cut-specs are parsed in a very simple way with the Python
standard library `shlex`, so if a filter contains spaces or other special
characters then it needs to be enclosed in single-quotes or double-quotes.
