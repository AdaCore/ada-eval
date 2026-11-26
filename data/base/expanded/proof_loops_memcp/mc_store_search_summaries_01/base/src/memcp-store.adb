with SPARK.Containers.Types;

with Sqlite_Vec_Spark;

package body Memcp.Store
  with SPARK_Mode => On
is

   package Sql renames Sqlite_Vec_Spark;

   --  Operators on the vectors' Capacity_Range, for the Length and Last_Count
   --  comparisons below.
   use type SPARK.Containers.Types.Count_Type;

   -----------------------------
   -- Checked statement layer --
   -----------------------------

   Sql_Error : exception;
   --  A SQLite status that is a failure rather than an answer. An operation
   --  catches it once, at the frame that owns the transaction. A non-Ok status
   --  that is an answer -- no such row, the end of a result set -- comes back
   --  as an out parameter instead, and never reaches here.

   procedure Prepare (S : Store; Text : String; Stmt : out Sql.Statement)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Is_Open (S) and then Text'Length > 0,
     Post              => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => Sql.Is_Reclaimed (Stmt));
   --  Compile Text against S. A statement that comes back valid belongs to the
   --  frame that asked for it, and that frame releases it in its `finally`
   --  part: only a raise from Prepare itself reclaims.

   procedure Bind (Stmt : Sql.Statement; Index : Positive; Value : Row_Id)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Bind a rowid at the 1-based Index.

   procedure Reset (Stmt : Sql.Statement)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Return a stepped statement to its initial state, bindings intact.

   procedure Step_Row (Stmt : Sql.Statement; Have_Row : out Boolean)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Advance a query one row. Have_Row is False at the end of the result set,
   --  which is the one non-Ok outcome that is an answer.

   --------------------
   -- Filter helpers --
   --------------------

   function Contains (L : Name_List; Value : String) return Boolean
   with Import, Global => null;
   --  Whether Value is one of the names in L, the Ada-side membership test
   --  for the search metadata filters. A full scan; filter lists are tiny.

   Fetch_Summary_SQL : constant String :=
     "SELECT s.id, p.name, s.session_id, s.created_at, s.header,"
     & " s.body, s.kind FROM summaries s"
     & " JOIN projects p ON p.id = s.project_id WHERE s.id = ?";
   --  One summaries row by id, with its project name. Also driven per
   --  candidate
   --  by Search_Summaries.

   ---------------------
   -- Search_Summaries --
   ---------------------

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
   is separate;

end Memcp.Store;
