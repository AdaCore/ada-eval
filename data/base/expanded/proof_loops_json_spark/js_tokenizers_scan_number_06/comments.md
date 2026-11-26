Loop 6 of `JSON.Tokenizers.Scan_Number` comes from `json/src/json-tokenizers.adb`, lines 579-587.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 2 checks unproved.

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

### `src/json-tokenizers.ads`

The subject loop is the sixth and last loop of Scan_Number: the one that applies a positive exponent to an integer mantissa by repeated multiplication by ten, bailing out with Error_Number_Out_Of_Range when the next multiplication would overflow.

Scan_Number scans a whole JSON number and holds six loops in sequence; an example contains one.  So this is not Scan_Number reduced but the sixth loop lifted out of it, under the name Scan_Number_Integer_Exponent, with the state the earlier loops left it as parameters and that loop's own invariant as the precondition.  Everything else of Scan_Number -- the stream, the fraction and exponent scanners, the float accumulator, the Token it builds -- is gone, because this loop reads none of it.

Int_Acc is accumulated as a non-positive value by Scan_Number, so that Base_Integer'First is reachable; that is the whole content of the precondition, and of the loop's invariant.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
