Loop 3 of `JSON.Tokenizers.Scan_Number` comes from `json/src/json-tokenizers.adb`, lines 473-500.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 6 loop invariants leaves 3 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- Renamed the unit in json-spark’s `proof_harness.ads` to `Instances` and placed it in `instances.ads`.
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

The subject loop is the third of Scan_Number's six loops: the one that consumes the digits of the exponent, accumulating them into Exp_Value and *saturating* at Exp_Limit rather than overflowing -- beyond that value the result is the same (zero, or out of range) for every larger exponent, because the mantissa has at most Maximum_Number_Length digits.

Scan_Number scans a whole JSON number and holds six loops in sequence; an example contains one.  So this is not Scan_Number reduced but its third loop lifted out, under the name Scan_Number_Exponent_Digits, with the state the earlier loops and the 'e' / sign handling left it as parameters and that loop's own invariant as the precondition.  Everything else of Scan_Number -- the integer and fraction scanners, the float conversion, the Token it builds -- is gone, because this loop reads none of it.

The precondition is the loop's own invariant: the stream has no buffered character, the number is not already at its maximum length, the two accumulators are within the range Add_Digit maintains, and the exponent has not yet passed its saturation point.

Int_Acc and F_Acc are `in` parameters and no statement of the loop touches them: the loop carries the accumulator bound as its third invariant all the same, because Scan_Number needs it after the loop, and copying the loop verbatim means keeping that invariant and therefore the two variables it names.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
