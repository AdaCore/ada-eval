--  memcp's storage layer over the sqlite_vec_spark primitives: the schema, the
--  record types the tools hand back, and the queries behind them. One SQLite
--  file, one vec0 table per embedded record type.

with SPARK.Containers.Formal.Unbounded_Vectors;

package Memcp.Store
  with SPARK_Mode => On
is

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

end Memcp.Store;
