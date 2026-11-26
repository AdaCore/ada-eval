Loop 1 of `JSON.Parsers.Parse_Object` comes from `json/src/json-parsers.adb`, lines 312-422.
The source revision is `fc76cda88dc08ddcb6923489e3af308bb935b820` in https://github.com/manthonyaiello/json-spark. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`. Some declarations or contracts have been adapted for proof in isolation.

The reference solution proves. In `base`, removing 8 loop invariants and the `:= null` initialization of `Element` (only an invariant reads it) leaves 7 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. The edits below also apply to both versions.

- Shortened the `----` rulers from 80 to 79 columns.
- Added a separate declaration for `To_Text`.
- In `json-types.ads`, `pragma Warnings` suppresses 'use clause for type "Text_Access" ... has no effect' (its body needs the clause).
- Moved `with JSON.Streams` into the body of `json-parsers.ads`, its only user.
- The comment that says which project GNATprove runs on names `main.gpr` instead of upstream's `json_prove.gpr`, which this project does not include.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/json-parsers.ads`

The visible part is empty here: Parse_Object is declared in the package *body*, as upstream declares it, and everything upstream exports -- the Parser type, Create, Create_From_File, Parse, Has_Storage, Destroy and Parse_Error -- is dropped, because none of it is on the path to the subject loop.  What survives is the generic contract itself, which is the point: GNATprove analyses only instances of a generic, so the loop has to reach it through an instantiation (see proof_harness.ads).

Check_Duplicate_Keys is kept, and it is the formal this example is about: it selects the branch of Parse_Object's loop that calls Types.Contains, and it is the only instance in which that call's precondition has to be discharged from the loop's invariants rather than short-circuited away. Default_Maximum_Depth is dropped with Create, the only subprogram that read it.

pragma Elaborate_Body is authored: a package whose visible part requires no completion does not permit a body, and the body is where the subject loop lives.

### `src/json-streams.ads`

The Stream the subject loop reads through, kept as upstream declares it: the full view, its Dynamic_Predicate and the three query expression functions are copied character for character, because their bodies *are* their meaning -- five of the loop's eight invariants and both clauses of its Loop_Variant are written in Length, Position and Has_Buffered_Character, and the predicate is what keeps Position's own postcondition (Position <= Length + 1) true, hence the variant expression Length - Position in range.

### `src/json-tokenizers.ads`

The tokenizer the subject loop drives.  Token_Kind, Token and Token_Status are copied whole; Scan_Token is imported as a spec with its contract character for character, and its progress clause is what the loop's Loop_Variant rests on.  Dropped: Error_Message, Read_Token and Tokenizer_Error, none of which Parse_Object calls.  The whole body of the package -- Scan_String, Scan_Number, Scan_Literal, Scan_Hex_Quad and Scan_Token itself -- is gone; SPARK is modular, so a callee's body is never needed to prove its caller.
