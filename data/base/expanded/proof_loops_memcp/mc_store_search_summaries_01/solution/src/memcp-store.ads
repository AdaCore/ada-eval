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

   Max_Search_Limit : constant := 1000;
   --  Ceiling on a search's requested result count. The KNN over-fetch factor
   --  is applied to the clamped value, so this also bounds the candidate scan.
   --  A larger request is clamped, not rejected.

   ------------------
   -- Record types --
   ------------------

   type Summary_Hit
     (Project_Len : Natural;   --  Length of Project.
      Session_Len : Natural;   --  Length of Session.
      Created_Len : Natural;   --  Length of Created_At.
      Header_Len  : Natural;   --  Length of Header.
      Body_Len    : Natural;   --  Length of Content.
      Kind_Len    : Natural)
   is   --  Length of Kind.
   record
      Id          : Row_Id;    --  The summary's rowid.
      Has_Session : Boolean;   --  session_id IS NOT NULL.
      Project     : String (1 .. Project_Len);     --  Project name.
      Session     : String (1 .. Session_Len);     --  Session id ("" if none).
      Created_At  : String (1 .. Created_Len);     --  ISO-8601 creation time.
      Header      : String (1 .. Header_Len);      --  The Header line.
      Content     : String (1 .. Body_Len);        --  The summary body.
      Kind        : String (1 .. Kind_Len);         --  Header kind.
      Distance    : Interfaces.IEEE_Float_64;      --  vec0 KNN L2 distance.
   end record;
   --  A summary search hit: a Summary's fields, flattened, plus the vec0 KNN
   --  L2
   --  distance. Smaller distance == closer; hits come back in
   --  ascending-distance
   --  order.

   package Summary_Hit_Vectors is new
     SPARK.Containers.Formal.Unbounded_Vectors
       (Index_Type   => Positive,
        Element_Type => Summary_Hit);
   --  SPARKlib vector instance over Summary_Hit.

   subtype Summary_Hit_List is Summary_Hit_Vectors.Vector;
   --  A list of summary search hits, as returned by Search_Summaries.

   procedure Search_Summaries
     (S         : Store;
      Projects  : Name_List;
      Limit     : Natural;
      Has_Since : Boolean;
      Since     : String;
      Has_Until : Boolean;
      Until_At  : String;
      Result    : out Summary_Hit_List;
      Status    : out Op_Status)
   with Pre => Is_Open (S);
   --  KNN search over summary embeddings, with the metadata filters applied on
   --  the Ada side to an over-fetched candidate set. Result is empty on
   --  Db_Error.
   --  @param S The open store to read.
   --  @param Projects The project-name filter (empty means every project).
   --  @param Limit The maximum number of hits to return.
   --  @param Has_Since Whether Since bounds the created_at window below.
   --  @param Since Inclusive lower ISO-8601 bound (when Has_Since).
   --  @param Has_Until Whether Until_At bounds the created_at window above.
   --  @param Until_At Inclusive upper ISO-8601 bound (when Has_Until).
   --  @param Result The matching hits, in ascending-distance order.
   --  @param Status Success, or Db_Error on a SQLite failure.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Keeps the body's query logic in SPARK.

   type Store is limited null record;
   --  Opaque here; see the header comment.

end Memcp.Store;
