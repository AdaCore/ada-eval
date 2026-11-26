Loop 1 of `Memcp.Extractor.Valid_Utf8` comes from `src/memcp-extractor.adb`, lines 78-139.
The source revision is `8701f4902513b6e2d350c179ba24a883993323c7` in https://github.com/manthonyaiello/memcp. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 5 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/memcp-extractor.ads`

Upstream Valid_Utf8 is declared and defined in the body of Memcp.Extractor, with no contract, so GNATprove inlines it into its single caller Decode_Base64 and reports it as "proved (0 checks)" -- its obligations are counted against the caller.  Here it is promoted to the spec, which is what makes it analysed on its own.  Promotion generalises the one call site to every String, so the example owes at least what upstream owed.
