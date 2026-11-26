--  The checked statement layer (Check / Prepare / Bind / Step_Row) is upstream
--  body-local; here each is imported with its upstream contract, since SPARK
--  needs no callee body to prove a caller.  Project_Id is imported for the
--  same reason.

with Sqlite_Vec_Spark;

package body Memcp.Store
  with SPARK_Mode => On
is

   package Sql renames Sqlite_Vec_Spark;

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

   procedure Bind (Stmt : Sql.Statement; Index : Positive; Value : String)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Bind text at the 1-based Index.

   procedure Bind (Stmt : Sql.Statement; Index : Positive; Value : Row_Id)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Bind a rowid at the 1-based Index.

   procedure Step_Row (Stmt : Sql.Statement; Have_Row : out Boolean)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Sql.Is_Valid (Stmt),
     Exceptional_Cases => (Sql_Error => True);
   --  Advance a query one row. Have_Row is False at the end of the result set,
   --  which is the one non-Ok outcome that is an answer.

   procedure Project_Id (S : Store; Name : String; Id : out Row_Id)
   with
     Import,
     Always_Terminates,
     Global            => (In_Out => Sql.DBMS),
     Pre               => Is_Open (S),
     Exceptional_Cases => (Sql_Error => True);
   --  The id of the project named Name, inserting the projects row when it
   --  does not exist yet.

   ------------------
   -- Save_Session --
   ------------------

   procedure Save_Session
     (S          : Store;
      Project    : String;
      Session_Id : String;
      Result     : out Session_Save_Result;
      Status     : out Op_Status)
   is separate;

end Memcp.Store;
