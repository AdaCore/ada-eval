Loop 5 of `JSON.Tokenizers.Scan_Number` comes from `json/src/json-tokenizers.adb`, lines 541-552.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 3 loop invariants and 1 assertion leaves 4 checks unproved.

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

The subject loop is the fifth of Scan_Number's six loops: the one that applies a *negative* decimal scale to a float mantissa.  It does not divide once per decimal place; it builds the divisor P up by tens and divides only when P is about to leave the exactly-representable range, so that a short decimal such as 3.14 is converted with a single rounding.

Scan_Number scans a whole JSON number and holds six loops in sequence; an example contains one.  So this is not Scan_Number reduced but its fifth loop lifted out, under the name Scan_Number_Scale_Down, with the state the earlier loops left it as parameters and that loop's own invariant as the precondition.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
