Loop 1 of `Memcp.Json.Get_Names` comes from `src/memcp-json.adb`, lines 219-233.
The source revision is `8701f4902513b6e2d350c179ba24a883993323c7` in https://github.com/manthonyaiello/memcp. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.
