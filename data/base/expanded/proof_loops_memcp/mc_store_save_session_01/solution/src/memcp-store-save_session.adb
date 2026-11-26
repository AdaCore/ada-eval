separate (Memcp.Store)
procedure Save_Session
  (S          : Store;
   Project    : String;
   Session_Id : String;
   Result     : out Session_Save_Result;
   Status     : out Op_Status)
is
   Proj_Id : Row_Id;
begin
   Result :=
     (Session_Row_Id  => 0,
      Chunk_Count     => 0,
      Already_Existed => False,
      Raw_Path_Set    => False);

   Project_Id (S, Project, Proj_Id);

   --  ---- idempotency: existing (project, session) row is a no-op ----
   declare
      Found : Boolean;
      Ex_Id : Row_Id := 0;
   begin
      Look_Up : declare
         Q : Sql.Statement;
      begin
         Prepare
           (S,
            "SELECT id FROM sessions WHERE project_id = ?"
            & " AND session_id = ?",
            Q);
         Bind (Q, 1, Proj_Id);
         Bind (Q, 2, Session_Id);
         Step_Row (Q, Found);
         if Found then
            Ex_Id := Sql.Column_Int64 (Q, 0);
         end if;
      finally
         Sql.Finalize (Q);
      end Look_Up;

      if Found then
         --  Return the existing row's id and its current chunk count,
         --  inserting nothing.
         Count_Chunks : declare
            C        : Sql.Statement;
            Have_Row : Boolean;
            Cnt      : Natural := 0;
         begin
            Prepare
              (S,
               "SELECT id FROM chunks WHERE session_row_id = ?"
               & " ORDER BY ordinal",
               C);
            Bind (C, 1, Ex_Id);
            loop
               pragma Loop_Invariant (Sql.Is_Valid (C));
               --  The frame's `finally` writes C; see Recent_Headers.
               Step_Row (C, Have_Row);
               exit when not Have_Row;
               --  A session holding more chunks than Natural can count is
               --  one this code cannot report on.
               if Cnt = Natural'Last then
                  raise Sql_Error;
               end if;
               Cnt := Cnt + 1;
            end loop;
            Result :=
              (Session_Row_Id  => Ex_Id,
               Chunk_Count     => Cnt,
               Already_Existed => True,
               Raw_Path_Set    => False);
            Status := Success;
         finally
            Sql.Finalize (C);
         end Count_Chunks;
         return;
      end if;
   end;

   Status := Success;
exception
   when Sql_Error =>
      Status := Db_Error;
end Save_Session;
