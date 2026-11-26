Loop 1 of `JSON.Types.Unescape` comes from `json/src/json-types.adb`, lines 119-195.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 2 loop invariants leaves 8 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. `Append_Code_Point` was given a separate spec carrying its contract.
