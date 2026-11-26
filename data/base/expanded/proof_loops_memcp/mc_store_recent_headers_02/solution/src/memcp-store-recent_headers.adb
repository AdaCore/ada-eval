separate (Memcp.Store)
procedure Recent_Headers
  (S        : Store;
   Projects : Name_List;
   N        : Natural;
   Result   : out Header_List;
   Status   : out Op_Status)
is
   Len_CT : constant Name_Vectors.Capacity_Range :=
     Name_Vectors.Length (Projects);
begin
   Result := Header_Vectors.Empty_Vector;

   --  No projects, or a filter too long to spell as a bounded IN clause:
   --  both return an empty list with Success.
   if Len_CT = 0 or else Len_CT > Max_Filter_Terms then
      Status := Success;
      return;
   end if;

   declare
      K        : constant Positive := Positive (Len_CT);
      Query    : constant String :=
        "SELECT s.id, p.name, s.session_id, s.created_at,"
        & " s.header, s.kind FROM summaries s"
        & " JOIN projects p ON p.id = s.project_id"
        & " WHERE p.name IN ("
        & Placeholders (K)
        & ")"
        & " ORDER BY s.created_at DESC LIMIT ?";
      Stmt     : Sql.Statement;
      Have_Row : Boolean;
   begin
      Prepare (S, Query, Stmt);

      --  Upstream sends the K project names to params 1 .. K here, in a
      --  loop of its own; that loop is mc_store_recent_headers_01.  N goes
      --  to the LIMIT param at K + 1.
      Bind (Stmt, K + 1, Row_Id (N));

      --  One Header_Entry per row; the Length guard discharges Append's
      --  capacity precondition.
      loop
         pragma Loop_Invariant (Sql.Is_Valid (Stmt));
         --  The enclosing frame's `finally` writes Stmt, which puts it in
         --  this loop's frame: without the invariant, Prepare's Post does
         --  not survive the back edge.
         Step_Row (Stmt, Have_Row);
         exit when not Have_Row;
         exit when
           Header_Vectors.Length (Result) = Header_Vectors.Last_Count;
         declare
            Id_C   : constant Row_Id := Sql.Column_Int64 (Stmt, 0);
            Proj   : Sql.Text_Ptr := Sql.Column_Text (Stmt, 1);
            Sess   : Sql.Text_Ptr := Sql.Column_Text (Stmt, 2);
            Crea   : Sql.Text_Ptr := Sql.Column_Text (Stmt, 3);
            Head   : Sql.Text_Ptr := Sql.Column_Text (Stmt, 4);
            Kind   : Sql.Text_Ptr := Sql.Column_Text (Stmt, 5);
            Null_S : constant Boolean := Sql.Column_Is_Null (Stmt, 2);
            Has_S  : constant Boolean := not Null_S;
            --  Two steps, as in Fetch_Summary: a volatile column read may
            --  not be an operand of `not`.
         begin
            Header_Vectors.Append
              (Result,
               Header_Entry'
                 (Project_Len => Proj.all'Length,
                  Session_Len => Sess.all'Length,
                  Created_Len => Crea.all'Length,
                  Header_Len  => Head.all'Length,
                  Kind_Len    => Kind.all'Length,
                  Summary_Id  => Id_C,
                  Has_Session => Has_S,
                  Project     => Proj.all,
                  Session     => Sess.all,
                  Created_At  => Crea.all,
                  Header      => Head.all,
                  Kind        => Kind.all));
            Sql.Free (Proj);
            Sql.Free (Sess);
            Sql.Free (Crea);
            Sql.Free (Head);
            Sql.Free (Kind);
         end;
      end loop;

      --  Leaving the loop with a row still pending is the vector filling
      --  up: the caller would be handed a truncated list, which is not a
      --  success.
      Status := (if Have_Row then Db_Error else Success);
   finally
      Sql.Finalize (Stmt);
   end;
exception
   when Sql_Error =>
      Status := Db_Error;
end Recent_Headers;
