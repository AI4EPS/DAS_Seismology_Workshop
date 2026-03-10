# DAS_db

`DAS_db` is a tool to generate a data catalog ("database") from a set of DAS
data files.  The catalog is in the form of a text file, where column values
are separated by spaces, and values with spaces are enclosed with
double-quotes.  This database file is consumed by other tools in this
codebase.

## Database Files

The database file records key details about _contiguous runs_ of samples.
(A single DAS data file may contain multiple contiguous runs, if the
recording unit happened to drop samples for some reason.)  For each
contiguous run of samples, the file has a row specifying these columns, as
indicated in the header row:

* `system` is the name of the DAS system or array.  For example,
  "Ridgecrest_South" or "Mammoth_North".

* `file` is the full absolute path to the DAS data file.

* `nSamples` is the number of samples in the contiguous run.

* `fs` is the sample frequency in Hz.

* `Desample` is a temporal decimation factor.  For example, if the original
  sample rate is 1000Hz, and the desample value is 5, then the actual sample
  rate is 200Hz.

* `startTime` is the ISO8601 date/time string for the first sample in the
  contiguous run.

* `endTime` is the ISO8601 date/time string for the last sample in the
  contiguous run.  (Note that the time of the next sample would be at
  `endTime + Desample/fs`.)

* `nChannels` is the number of channels (i.e. points along the optical fiber)
   in the DAS data.

* `dCh` is the spatial distance between channels along the fiber, typically
  reported in meters.

* `GaugeLen` is the length of the fiber that each channel's strain
  measurement is measured over, centered around the channel location.  This
  value is typically reported in meters.

* `firstSample` is the index of this contiguous run's first sample in the
  data file.  The first sample in the file has an index of 0.  This value
  will be nonzero if the data file contains multiple contiguous runs.

Note that in some cases, where a value is not available for a given data
file, the corresponding field may be set to `-1` instead.

## DAS_db Command-Line Arguments

The `DAS_db` program can be used to either generate a database catalog file
from scratch, or to update an existing catalog file.  The general form of the
command is:

    python DAS_db.py "<file-glob-pattern>" dasdb.txt "DAS_System_Name" [OPTIONS]

There are three required arguments which must be specified in this order.
(Options may precede or follow these arguments.)

The `file-glob-pattern` argument specifies a glob pattern (e.g. including the
`*` and `?` characters) to find DAS data files that should be scanned for the
database.  The program finds files that match this glob-pattern using the
[standard Python library `glob`](https://docs.python.org/3/library/glob.html).
**Note that the program doesn't rely on the command-shell to expand this glob
pattern.**  This means that the glob should be quoted so the shell will not
expand it.

The `dasdb.txt` file is the database file updated by this command.  By
default, if the file already exists, it will be read at the start of
execution and will be updated.  Files that already appear in the database
file will be skipped.  (See the `--overwrite` argument below if you wish to
modify the default behavior.)

The `"DAS_System_Name"` argument specifies the name of the DAS array.  For
example, it might be `"Ridgecrest_South"` or `"Mammoth_North"`.

### Options

Here are some options that can be passed to `DAS_db`:

* `-v` or `--verbose` can be specified to turn on verbose output

* `--overwrite` causes `DAS_db` to overwrite the database file if it already
  exists.  **The default behavior is to merely load and update the database
  file if it already exists.**  This switch disables this behavior.

* `-nTh` or `--nThreads` can be specified to tell the program how many
  concurrent threads should be used to scan data files.  Since this is an
  IO-bound operation, multiple threads of execution tends to make this much
  faster.  The default value is the number of CPU cores on the system.

## Example

Here is an example invocation of the `DAS_db` script:

    python DAS_db.py -v "das-data/ridgecrest-*.h5" ridgecrest-db.txt "Ridgecrest_South"

This will find all `.h5` files that match the specified glob-pattern, and
will update (or create) the `ridgecrest-db.txt` database file.  The DAS
system name is `Ridgecrest_South`.

Verbose output is also enabled, so that you get more information about what
the program is doing.