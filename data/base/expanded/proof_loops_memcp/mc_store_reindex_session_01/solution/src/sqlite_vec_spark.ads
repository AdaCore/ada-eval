--  Extracted from memcp, crates/sqlite_vec_spark/src/sqlite_vec_spark.ads.
--
--  The abstract state is kept, because the loop writes it (Step advances a
--  cursor).  With no body here to initialize it, the package carries
--  pragma Elaborate_Body and an empty SPARK_Mode => Off body, as the
--  extraction recipe prescribes for that case.

with Ada.Unchecked_Deallocation;
with Interfaces;

package Sqlite_Vec_Spark
  with
    SPARK_Mode     => On,
    Abstract_State =>
      (DBMS with
        External =>
          (Async_Writers    => True,
           Async_Readers    => True,
           Effective_Writes => True,
           Effective_Reads  => False)),
    --  model the SQLite subsystem's state on the far side of the C
    --  boundary.
    Initializes    => DBMS
is

   pragma Elaborate_Body;

   type Statement is limited private
   with
     Annotate                  => (GNATprove, Ownership, "Needs_Reclamation"),
     Default_Initial_Condition =>
       not Is_Valid (Statement) and then Is_Reclaimed (Statement);
   --  Opaque prepared-statement handle.

   type Text_Ptr is access String;
   --  Caller-owned text pulled from a column, allocated to exactly the
   --  column's
   --  byte length; the caller frees it with Free. gnatprove tracks the
   --  ownership, so a dropped Text_Ptr is a proof error, not a silent leak.

   procedure Free is new Ada.Unchecked_Deallocation (String, Text_Ptr);
   --  Reclaim a Text_Ptr returned by Column_Text.

   function Is_Valid (S : Statement) return Boolean;
   --  True when S names a live (not yet finalized) prepared statement.
   --  @param S The statement handle to test.
   --  @return True iff S is valid.

   function Is_Reclaimed (S : Statement) return Boolean
   with Ghost, Annotate => (GNATprove, Ownership, "Is_Reclaimed");
   --  Reclamation predicate for Statement: a finalized statement holds no C
   --  resource.
   --  @param S The statement handle to test.
   --  @return True iff S owns no statement (equivalently, not Is_Valid (S)).

   procedure Finalize (S : in out Statement)
   with
     Import,
     Always_Terminates,
     Post    => not Is_Valid (S) and then Is_Reclaimed (S),
     Global  => (In_Out => DBMS),
     Depends => (DBMS =>+ null, S => null, null => S);
   --  Destroy the statement (sqlite3_finalize). Idempotent; leaves S
   --  not-valid.
   --  The explicit Depends -- S's new value is a constant, its old handle
   --  reaching only the C side -- keeps callers that finalize a local
   --  Statement
   --  free of "set but not used" flow warnings.
   --  @param S The statement to finalize; left not-valid.

   ------------------
   -- Column reads --
   ------------------

   function Column_Int64
     (S : Statement; Col : Natural) return Interfaces.Integer_64
   with
     Import,
     Pre    => Is_Valid (S),
     Volatile_Function,
     Global => (Input => DBMS);
   --  Read column Col (0-based) of the current row as a 64-bit integer. Every
   --  Column_* read is valid only when the most recent Step returned Row, and
   --  only until the next Step/Reset/Finalize on S -- unenforceable in SPARK.
   --  @param S The valid statement positioned on a result row.
   --  @param Col The 0-based column index.
   --  @return The column value as an Integer_64.

   function Column_Double
     (S : Statement; Col : Natural) return Interfaces.IEEE_Float_64
   with
     Import,
     Pre    => Is_Valid (S),
     Volatile_Function,
     Global => (Input => DBMS);
   --  Read column Col of the current row as a double; the vec0 KNN `distance`
   --  column comes back here.
   --  @param S The valid statement positioned on a result row.
   --  @param Col The 0-based column index.
   --  @return The column value as an IEEE_Float_64.

   function Column_Is_Null (S : Statement; Col : Natural) return Boolean
   with
     Import,
     Pre    => Is_Valid (S),
     Volatile_Function,
     Global => (Input => DBMS);
   --  True when the column holds SQL NULL -- the only way to tell that from an
   --  empty-string Column_Text result.
   --  @param S The valid statement positioned on a result row.
   --  @param Col The 0-based column index.
   --  @return True iff the column holds SQL NULL.

   function Column_Text (S : Statement; Col : Natural) return Text_Ptr
   with
     Import,
     Pre    => Is_Valid (S),
     Post   => Column_Text'Result /= null,
     Volatile_Function,
     Global => (Input => DBMS);
   --  A fresh, exactly-sized, caller-owned copy of the column's text, which
   --  the
   --  caller must Free -- a copy because sqlite3_column_text's buffer belongs
   --  to
   --  SQLite and dies at the next cursor move. Never null; a NULL or empty
   --  column yields "" (length 0).
   --  @param S The valid statement positioned on a result row.
   --  @param Col The 0-based column index.
   --  @return A fresh, caller-owned copy of the column's text; never null.

private

   pragma Annotate (GNATprove, Hide_Info, "Private_Part");
   --  Required for GNATprove Ownership

   --  The raw SQLite statement pointer, an ownership type of its own so the
   --  Bridge cannot pass a connection to a statement's release operation or
   --  vice versa.
   package Handles is

      type Stmt_Handle is private
      with
        Default_Initial_Condition => Is_Null (Stmt_Handle),
        Annotate                  =>
          (GNATprove, Ownership, "Needs_Reclamation"),
        Annotate                  =>
          (GNATprove, Predefined_Equality, "Only_Null");
      --  The C sqlite3_stmt*. Owns a compiled statement until it is finalized.

      Null_Stmt_Handle : constant Stmt_Handle
      with
        Annotate => (GNATprove, Ownership, "Reclaimed_Value"),
        Annotate => (GNATprove, Predefined_Equality, "Null_Value");
      --  The reclaimed value: a Stmt_Handle equal to this owns nothing, so
      --  GNATprove permits dropping or overwriting it.

      function Is_Null (H : Stmt_Handle) return Boolean
      with
        Ghost,
        Global => null,
        Post   => Is_Null'Result = (H = Null_Stmt_Handle);
      --  Ghost spelling of "reclaimed", for the Default_Initial_Condition
      --  above; executable code compares directly.
      --  @param H The handle to test.
      --  @return True iff H is the reclaimed value.

   private
      pragma SPARK_Mode (Off);

      type Sqlite3_Stmt is limited null record;
      --  Designated type for the C sqlite3_stmt. Never allocated or
      --  dereferenced on the Ada side: every value comes from SQLite and goes
      --  back to it, so all the representation owes us is a pointer comparable
      --  to null.

      type Stmt_Handle is access all Sqlite3_Stmt;
      --  The C sqlite3_stmt*, owned until finalized; full view a plain C
      --  pointer.

      Null_Stmt_Handle : constant Stmt_Handle := null;
      --  The reclaimed Stmt_Handle value: the null pointer.

      function Is_Null (H : Stmt_Handle) return Boolean
      is (H = null);
      --  Completion of the Stmt_Handle ghost predicate.
      --  @param H The handle to test.
      --  @return True iff H is the null pointer.
   end Handles;

   use type Handles.Stmt_Handle;

   type Statement is limited record
      Handle : Handles.Stmt_Handle;
      --  The owned sqlite3_stmt*; Null_Stmt_Handle (the default) when invalid.
   end record;
   --  Opaque prepared-statement handle.

   function Is_Valid (S : Statement) return Boolean
   is (S.Handle /= Handles.Null_Stmt_Handle);
   --  A statement is valid iff its handle is not the reclaimed value.
   --  @param S The statement handle to test.
   --  @return True iff S holds a compiled statement.

   function Is_Reclaimed (S : Statement) return Boolean
   is (S.Handle = Handles.Null_Stmt_Handle);
   --  Reclaimed exactly when the handle is the reclaimed value (equivalently,
   --  not Is_Valid (S)).
   --  @param S The statement handle to test.
   --  @return True iff S owns no statement.

end Sqlite_Vec_Spark;
