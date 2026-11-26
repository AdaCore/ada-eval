--  Extracted from memcp, src/memcp-store.ads and .adb.

with Interfaces;

with SPARK.Containers.Formal.Unbounded_Vectors;


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

   ----------------------
   -- List parameters  --
   ----------------------

   type Name (Len : Natural) is record
      Value : String (1 .. Len);   --  The name text.
   end record;
   --  A single variable-length name (project name or session id), indefinite
   --  so
   --  a list of them can live in a SPARKlib vector. Project and session names
   --  are UNIQUE, so a filter binds the name straight into `WHERE ... IN (?,
   --  ?,
   --  ...)` instead of resolving it to an id first.
   --  @field Len Length of the name text.

   package Name_Vectors is new
     SPARK.Containers.Formal.Unbounded_Vectors
       (Index_Type   => Positive,
        Element_Type => Name);
   --  SPARKlib vector instance over Name, for name-filter lists.

   subtype Name_List is Name_Vectors.Vector;
   --  A list of names (project or session-id filter terms).

   Max_Filter_Terms : constant := 1024;
   --  Upper bound on how many terms a dynamically built `IN (...)` clause may
   --  carry. Keeps the placeholder string (2*N - 1 characters) well clear of
   --  Integer'Last, and sits far above any real filter. A longer filter is
   --  refused, not truncated.

   ------------------
   -- Record types --
   ------------------

   type Header_Entry
     (Project_Len : Natural;   --  Length of Project.
      Session_Len : Natural;   --  Length of Session.
      Created_Len : Natural;   --  Length of Created_At.
      Header_Len  : Natural;   --  Length of Header.
      Kind_Len    : Natural)
   is   --  Length of Kind.
   record
      Summary_Id  : Row_Id;    --  The summary's rowid.
      Has_Session : Boolean;   --  session_id IS NOT NULL.
      Project     : String (1 .. Project_Len);     --  Project name.
      Session     : String (1 .. Session_Len);     --  Session id ("" if none).
      Created_At  : String (1 .. Created_Len);     --  ISO-8601 creation time.
      Header      : String (1 .. Header_Len);      --  The Header line.
      Kind        : String (1 .. Kind_Len);        --  Header kind.
   end record;
   --  One Header, the unit Recent_Headers returns: a summaries row without its
   --  body, which is what fetch_summary is for.

   package Header_Vectors is new
     SPARK.Containers.Formal.Unbounded_Vectors
       (Index_Type   => Positive,
        Element_Type => Header_Entry);
   --  SPARKlib vector instance over Header_Entry.

   subtype Header_List is Header_Vectors.Vector;
   --  A list of Headers, as returned by Recent_Headers.

   procedure Recent_Headers
     (S        : Store;
      Projects : Name_List;
      N        : Natural;
      Result   : out Header_List;
      Status   : out Op_Status)
   with Pre => Is_Open (S);
   --  The N most recent Headers across the given Projects, newest first
   --  (created_at DESC). An empty Projects list, or one longer than
   --  Max_Filter_Terms, yields an empty Result with Success rather than an
   --  unbounded IN clause. Result is always initialized; on Db_Error it is
   --  empty.
   --  @param S The open store to read.
   --  @param Projects The project-name filter (empty yields an empty Result).
   --  @param N The maximum number of Headers to return.
   --  @param Result The matching Headers, newest first.
   --  @param Status Success, or Db_Error on a SQLite failure.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Keeps the body's query logic in SPARK.

   type Store is limited null record;
   --  Opaque here; see the header comment.

end Memcp.Store;
