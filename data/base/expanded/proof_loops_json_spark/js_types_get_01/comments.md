Loop 1 of `JSON.Types.Get` comes from `json/src/json-types.adb`, lines 231-240.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- Renamed the unit in json-spark’s `proof_harness.ads` to `Instances` and placed it in `instances.ads`.
- Shortened the `----` rulers from 80 to 79 columns.
- In `json-types.ads`, `pragma Warnings` suppresses 'use clause for type "Text_Access" ... has no effect' (its body needs the clause).
- The comment that says which project GNATprove runs on names `main.gpr` instead of upstream's `json_prove.gpr`, which this project does not include.
