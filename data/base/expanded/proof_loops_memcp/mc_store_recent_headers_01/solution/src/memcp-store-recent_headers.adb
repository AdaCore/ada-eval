separate (Memcp.Store)
procedure Recent_Headers
  (S : Store; Projects : Name_List; N : Natural; Status : out Op_Status)
is
   Len_CT : constant Name_Vectors.Capacity_Range :=
     Name_Vectors.Length (Projects);
begin
   --  No projects, or a filter too long to spell as a bounded IN clause:
   --  both return an empty list with Success.
   if Len_CT = 0 or else Len_CT > Max_Filter_Terms then
      Status := Success;
      return;
   end if;

   declare
      K     : constant Positive := Positive (Len_CT);
      Query : constant String :=
        "SELECT s.id, p.name, s.session_id, s.created_at,"
        & " s.header, s.kind FROM summaries s"
        & " JOIN projects p ON p.id = s.project_id"
        & " WHERE p.name IN ("
        & Placeholders (K)
        & ")"
        & " ORDER BY s.created_at DESC LIMIT ?";
      Stmt  : Sql.Statement;
   begin
      Prepare (S, Query, Stmt);

      --  The K project names go to params 1 .. K, the 1-based vector index
      --  doubling as the bind position, and N to the LIMIT param at K + 1.
      for I in
        Name_Vectors.First_Index (Projects)
        .. Name_Vectors.Last_Index (Projects)
      loop
         pragma Loop_Invariant (Sql.Is_Valid (Stmt));
         --  The enclosing frame's `finally` writes Stmt, which puts it in
         --  this loop's frame: without the invariant, Prepare's Post does
         --  not survive the back edge.
         Bind (Stmt, I, Name_Vectors.Element (Projects, I).Value);
      end loop;
      Bind (Stmt, K + 1, Row_Id (N));

      --  Upstream steps the statement here, building one Header_Entry per
      --  row; that loop is mc_store_recent_headers_02.
      Status := Success;
   finally
      Sql.Finalize (Stmt);
   end;
exception
   when Sql_Error =>
      Status := Db_Error;
end Recent_Headers;
