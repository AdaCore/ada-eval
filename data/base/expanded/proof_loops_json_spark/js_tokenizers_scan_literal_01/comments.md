Loop 1 of `JSON.Tokenizers.Scan_Literal` comes from `json/src/json-tokenizers.adb`, lines 641-655.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 4 loop invariants leaves 2 checks unproved.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/json-streams.ads`

The Stream abstraction the subject loop reads through, kept as upstream declares it: the full view, its Dynamic_Predicate and the four query functions are copied character for character, because they are expression functions whose bodies *are* their meaning -- the loop's invariants are written in terms of Length, Position and Has_Buffered_Character, and the predicate is what makes Position's own postcondition (Position <= Length + 1) hold.

### `src/json.ads`

Copied verbatim from json-spark, json/src/json.ads.  The subject loop's unit is a child of this package, so the parent comes with it.
