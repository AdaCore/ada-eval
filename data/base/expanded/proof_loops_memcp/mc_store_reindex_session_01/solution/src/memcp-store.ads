--  Extracted from memcp, src/memcp-store.ads and .adb.
--
--  Reindex_Session is upstream's Reindex_Session reduced to the loop that
--  drops the old chunk embeddings: the bulk chunk delete, the re-insert of the
--  replacement turns and the COMMIT that follows are gone, and with them the
--  Chunks parameter and the New_Count it reported.

with Interfaces;


package Memcp.Store
  with SPARK_Mode => On
is

   subtype Row_Id is Interfaces.Integer_64;
   --  A SQLite rowid: signed 64-bit.

   type Store is limited private;
   --  The storage handle: owns one SQLite connection, with vec0 registered,
   --  foreign_keys ON and WAL. Limited, because a copy would double-close that
   --  connection.

   function Is_Open (S : Store) return Boolean
   with Import, Global => null;
   --  Whether the store's connection is currently open.  Upstream this is an
   --  expression function over the owned connection handle; here it is opaque,
   --  because the loop's proof uses it only to discharge the wrappers' Pre.
   --  @param S The store to test.
   --  @return True iff S holds an open SQLite connection.

   type Op_Status is (Success, Db_Error);
   --  Generic outcome for the row-touching operations below.
   --  @enum Success The operation completed.
   --  @enum Db_Error A SQLite error surfaced from the primitives layer
   --    (prepare/step/commit failure).

   procedure Reindex_Session
     (S          : Store;
      Project    : String;
      Session_Id : String;
      Found      : out Boolean;
      Old_Count  : out Natural;
      Status     : out Op_Status)
   with
     Pre =>
       Is_Open (S)
       and then Project'Length > 0
       and then Project'Last < Natural'Last;
   --  Replace a stored session's chunks (+ embeddings) in place -- here
   --  reduced
   --  to the first half of that: locate the session row, then delete each
   --  existing chunk's embedding, vec0 having no FK cascade.  Found => False
   --  (with Success) when no session row exists for (Project, Session_Id);
   --  otherwise Old_Count reports how many embeddings were dropped.
   --  @param S The open store to modify.
   --  @param Project The project name.
   --  @param Session_Id The session id.
   --  @param Found True iff a session row existed for (Project, Session_Id).
   --  @param Old_Count Number of chunks present before the swap.
   --  @param Status Success, or Db_Error on a SQLite failure.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Keeps the body's query logic in SPARK.

   type Path_Access is access String;
   --  Owning handle for the opened DB path, remembered at Open so a session
   --  save can place raw transcripts under <db_parent>/sessions/... null
   --  before
   --  Open and after Close, which frees it.

   type Store is limited null record;
   --  Opaque here; see the header comment.

end Memcp.Store;
