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

   procedure Recent_Headers
     (S : Store; Projects : Name_List; N : Natural; Status : out Op_Status)
   with Pre => Is_Open (S);
   --  Bind the project-name filter of the "N most recent Headers" query: the
   --  K names to parameters 1 .. K and N to the LIMIT parameter at K + 1. An
   --  empty Projects list, or one longer than Max_Filter_Terms, is refused
   --  with Success rather than spelled as an unbounded IN clause.
   --  @param S The open store to read.
   --  @param Projects The project-name filter (empty binds nothing).
   --  @param N The maximum number of Headers to return.
   --  @param Status Success, or Db_Error on a SQLite failure.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Keeps the body's query logic in SPARK.

   type Store is limited null record;
   --  Opaque here; see the header comment.

end Memcp.Store;
