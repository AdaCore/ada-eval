Loop 1 of `JSON.Tokenizers.Scan_Token` comes from `json/src/json-tokenizers.adb`, lines 686-704.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 3 loop invariants leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/json-streams.ads`

The Stream abstraction the subject loop reads through, kept as upstream declares it: the full view, its Dynamic_Predicate and the four query functions are copied character for character, because they are expression functions whose bodies *are* their meaning -- the loop's invariants are written in terms of Length, Position and Has_Buffered_Character, and the predicate is what makes Position's own postcondition (Position <= Length + 1) hold.

### `src/json-tokenizers.ads`

Scan_String, Scan_Number and Scan_Literal are declared in the package body upstream and are imported here with their contracts unchanged, because the whole of Scan_Token's body is kept: the subject loop's three invariants exist to discharge the progress clause of the contract below, and truncating the body after the loop would leave the third one with nothing to prove.

Its guard cannot hold.  Read_Character returns with no buffered character, so the whitespace loop cannot leave one; the only paths that can are Scan_Number and Scan_Literal, and each promises `if Has_Buffered_Character (Stream) then Position (Stream) > Position (Stream)'Old`, while Position never decreases.  So a buffered character on return implies Position > Position'Old, the guard is false, and GNATprove reports `unreachable branch` on the `then` part under --proof-warnings=on.  The clause discharges no check: with it and without it the example proves the same 33 checks, 13 of them Scan_Token's, the count upstream reports.  Dropping it weakens what Scan_Token promises -- free here because the example has no callers -- rather than leaving dead proof code to read as proof content.  The progress clause below it, which the loop's three invariants do discharge, is untouched.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
