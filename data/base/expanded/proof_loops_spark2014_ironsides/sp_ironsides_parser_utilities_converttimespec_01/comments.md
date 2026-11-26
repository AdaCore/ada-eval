Loop 1 of `Parser_Utilities.ConvertTimeSpec` comes from `testsuite/gnatprove/tests/ironsides/parser_utilities.adb`, lines 881-910.
The source revision is `deb177c58ac4e7b0e761f00456910468646d9a43` in https://github.com/AdaCore/spark2014. The dataset records the upstream licence as `permissive-usafa`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 1 loop invariant leaves 5 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. `IsMult` and `MultValue` were given separate specs, the latter carrying its contract.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/rr_type-soa_record_type.ads`

Extracted from spark2014, testsuite/gnatprove/tests/ironsides. Trimmed to the one constant the subject loop's invariant refers to.
