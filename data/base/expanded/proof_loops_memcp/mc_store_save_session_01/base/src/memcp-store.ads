--  Extracted from memcp, src/memcp-store.ads and .adb.

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

   type Session_Save_Result is record
      Session_Row_Id  : Row_Id;    --  The sessions-row id.
      Chunk_Count     : Natural;   --  How many chunks are stored for it.
      Already_Existed :
        Boolean;   --  The (project, session_id) row already existed.
      Raw_Path_Set    :
        Boolean;   --  The raw transcript reached disk (raw_path set).
   end record;
   --  Save_Session's outcome: the sessions-row id, the stored chunk count, and
   --  the idempotency and transcript-write flags. Chunks are tallied, not
   --  enumerated -- no chunk ids come back.

   procedure Save_Session
     (S          : Store;
      Project    : String;
      Session_Id : String;
      Result     : out Session_Save_Result;
      Status     : out Op_Status)
   with
     Pre =>
       Is_Open (S)
       and then Project'Length > 0
       and then Project'Last < Natural'Last
       and then Session_Id'Last < Natural'Last;
   --  Insert a session row + its chunks (each with its embedding):
   --
   --    * Idempotent on (Project, Session_Id): if a row already exists, its id
   --      and current chunk count come back with Already_Existed => True and
   --      nothing is re-inserted or rewritten.
   --
   --  @param S The open store to write to.
   --  @param Project The project name.
   --  @param Session_Id The session id.
   --  @param Result The session id, chunk count, and idempotency/write flags.
   --  @param Status Success, or Db_Error on a SQLite failure.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Keeps the body's query logic in SPARK.

   type Store is limited null record;
   --  Opaque here; see the header comment.

end Memcp.Store;
