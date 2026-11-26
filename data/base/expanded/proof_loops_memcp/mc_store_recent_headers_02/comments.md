Loop 2 of `Memcp.Store.Recent_Headers` comes from `src/memcp-store.adb`, lines 1068-1107.
The source revision is `8701f4902513b6e2d350c179ba24a883993323c7` in https://github.com/manthonyaiello/memcp. The dataset records the upstream licence as `apache-2.0`.

The standalone project contains the subprogram and the declarations it uses. It declares some callees with `Import`.

The reference solution proves. In `base`, removing 1 loop invariant leaves 8 checks unproved.

## Changes to the upstream source

Both versions use gnatformat, with comment spacing, line breaks, identifier case
and `end` labels adjusted to satisfy GNAT’s build checks. `with Sqlite_Vec_Spark` was moved from `memcp-store.ads` to `memcp-store.adb`, the only unit that uses it.

## Notes moved out of the sources

These explanations were comments in the project sources. They describe how
the sample was built, so they live here instead of in the files the prompt
hands over.

### `src/memcp-store.adb`

Memcp.Store body, reduced to the subject loop: Recent_Headers' row loop, which steps the prepared statement and appends one Header_Entry per row.

The checked statement layer (Check / Prepare / Bind / Step_Row) is upstream body-local; here each is imported with its upstream contract, since SPARK needs no callee body to prove a caller.  Placeholders is imported for a second reason as well: its own body holds a loop, and an example carries exactly one.

### `src/memcp-store.ads`

Store is opaque here: the subject loop passes it only to the checked statement wrappers, which are imported, so the upstream representation (an owned Sqlite_Vec_Spark.Database plus the remembered DB path) is not needed to prove the loop.  Dropping the Ownership annotation and the reclamation predicate with it removes information the proof might have used, so it is a weakening.

Recent_Headers is upstream's Recent_Headers reduced to its SECOND loop, the one that steps the statement and builds one Header_Entry per row.  The bind loop that precedes it is a separate example (mc_store_recent_headers_01); everything else about the subprogram, including the guard and the query construction that reach this loop, is upstream's.

### `src/sqlite_vec_spark.ads`

Only what the subject loop touches directly: the prepared-statement handle with its ownership annotations, the column reads, and Finalize -- the operation the enclosing frame's `finally` part calls, which is what puts the handle in the loop's frame.  The connection handle (Database), the Status type and the bind/step primitives are gone: the loop reaches them only through Memcp.Store's own checked wrappers, which are imported with their upstream contracts in the body.
