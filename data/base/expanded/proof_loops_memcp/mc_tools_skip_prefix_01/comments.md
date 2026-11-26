Loop 1 of `Memcp.Tools.Skip_Prefix` comes from `src/memcp-tools.adb`, lines 264-268.
The source revision is `8701f4902513b6e2d350c179ba24a883993323c7` in https://github.com/manthonyaiello/memcp. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses.

The reference solution proves. In `base`, removing 1 loop invariant leaves 1 check unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/memcp-tools.ads`

Upstream it is a body-local function of Memcp.Tools, declared with this contract in the body's declarative part and defined further down; here it is declared in the spec so that it is the example's entry point.  The rest of Memcp.Tools -- the tool enumeration, the Do_* procedures, the JSON and Memcp.Store machinery -- is not needed: Skip_Prefix reads only its two parameters and calls only the expression function After, which is copied whole into the body below.
