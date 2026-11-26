Loop 2 of `JSON.Tokenizers.Scan_Number` comes from `json/src/json-tokenizers.adb`, lines 398-422.
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

The subject loop is the second of Scan_Number's six loops: the one that consumes the digits after the decimal point, adding each to the same pair of accumulators the integer part used and counting them in Frac_Digits, which the float conversion later subtracts from the exponent.

Scan_Number scans a whole JSON number and holds six loops in sequence; an example contains one.  So this is not Scan_Number reduced but its second loop lifted out, under the name Scan_Number_Fraction_Digits, with the state the integer scanner and the decimal-point handling left it as parameters and that loop's own invariant as the precondition.  Everything else of Scan_Number -- the integer and exponent scanners, the float conversion, the Token it builds -- is gone, because this loop reads none of it.

The precondition is the loop's own invariant: the stream has no buffered character, the number is not already at its maximum length, and the two accumulators are within the range Add_Digit maintains -- Int_Acc non-positive so that Base_Integer'First is reachable, F_Acc below the bound at which the mantissa is abandoned.  Frac_Digits < Char_Count + 1 is upstream's own spelling of "no more fraction digits than characters scanned".

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
