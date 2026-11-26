Loop 1 of `JSON.Tokenizers.Scan_Number` comes from `json/src/json-tokenizers.adb`, lines 334-360.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 6 loop invariants leaves 5 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- Renamed the unit in json-spark’s `proof_harness.ads` to `Instances` and placed it in `instances.ads`.
- Added a separate declaration for `Add_Digit` and placed its contract there.
- In `json-types.ads`, `pragma Warnings` suppresses 'type "Float_Type" is not referenced', 'type "Integer_Type" is not referenced'.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/instances.ads`

Copied from json-spark, json/prove/proof_harness.ads, reduced to the two instantiations this example needs.  GNATprove analyses only instances of a generic unit, and the subject loop lives in the body of the generic JSON.Tokenizers, so without this file the example would measure nothing.

### `src/json-streams.ads`

The Stream abstraction the subject loop reads through, kept as upstream declares it: the full view, its Dynamic_Predicate and the three query functions are copied character for character, because they are expression functions whose bodies *are* their meaning -- the loop's invariants are written in terms of Length, Position and Has_Buffered_Character, and the predicate is what makes Position's own postcondition (Position <= Length + 1) hold.

### `src/json-tokenizers.ads`

The subject loop is the first of Scan_Number's six loops: the one that consumes the digits of the integer part, accumulating them into an integer and a float mantissa at once and refusing a number longer than Maximum_String_Length_Numbers characters.

Scan_Number scans a whole JSON number and holds six loops in sequence; an example contains one.  This being the *first* of them, nothing has to be assumed about what an earlier loop left: Scan_Number's own precondition is the precondition here, upstream's prologue is kept as pre-loop code, and it is that prologue -- not a contract -- that establishes the loop's invariant on entry.  What follows the loop is truncated after the two tests that read the digit count, so Scan_Number's fraction and exponent scanners, its float conversion and the Token it builds are all gone.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
