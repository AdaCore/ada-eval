separate (Memcp.Store)
procedure Reindex_Session
  (S          : Store;
   Project    : String;
   Session_Id : String;
   Found      : out Boolean;
   Old_Count  : out Natural;
   Status     : out Op_Status)
is
   Proj_Id : Row_Id;
begin
   Found := False;
   Old_Count := 0;

   Project_Id (S, Project, Proj_Id);

   --  Locate the session row + its original created_at (copied out so the
   --  new chunks can inherit it after the cursor is gone). The copy doubles
   --  as the found flag: no row, no copy.
   declare
      Sess_Id : Row_Id := 0;
      TS_Copy : Path_Access := null;
   begin
      Look_Up : declare
         Q    : Sql.Statement;
         Have : Boolean;
      begin
         Prepare
           (S,
            "SELECT id, created_at FROM sessions"
            & " WHERE project_id = ? AND session_id = ?",
            Q);
         Bind (Q, 1, Proj_Id);
         Bind (Q, 2, Session_Id);
         Step_Row (Q, Have);
         if Have then
            Sess_Id := Sql.Column_Int64 (Q, 0);
            declare
               T : Sql.Text_Ptr := Sql.Column_Text (Q, 1);
            begin
               TS_Copy := new String'(T.all);
               Sql.Free (T);
            end;
         end if;
      finally
         Sql.Finalize (Q);
      end Look_Up;

      if TS_Copy = null then
         --  No such session: Found stays False, which is an answer.
         Status := Success;
      else
         --  Replace the chunks in one transaction: delete each old chunk's
         --  embedding (vec0 has no FK cascade), bulk-delete the chunk rows,
         --  then insert the new ones with the session's original timestamp.
         Exec (S, "BEGIN");

         Transaction : begin
            Drop_Vectors : declare
               Sel      : Sql.Statement;
               Have_Row : Boolean;
               Cnt      : Natural := 0;
            begin
               Prepare
                 (S, "SELECT id FROM chunks WHERE session_row_id = ?", Sel);
               Bind (Sel, 1, Sess_Id);
               loop
                  Step_Row (Sel, Have_Row);
                  exit when not Have_Row;
                  --  A session holding more chunks than Natural can count
                  --  is one this code cannot reindex.
                  if Cnt = Natural'Last then
                     raise Sql_Error;
                  end if;
                  declare
                     Old_Id : constant Row_Id := Sql.Column_Int64 (Sel, 0);
                     DV     : Sql.Statement;
                  begin
                     Prepare
                       (S, "DELETE FROM chunk_vec WHERE rowid = ?", DV);
                     Bind (DV, 1, Old_Id);
                     Step_Done (DV);
                  finally
                     Sql.Finalize (DV);
                  end;
                  Cnt := Cnt + 1;
               end loop;
               Old_Count := Cnt;
            finally
               Sql.Finalize (Sel);
            end Drop_Vectors;

            --  Upstream continues inside this transaction with the bulk
            --  chunk delete, the re-insert of the replacement turns, and
            --  the COMMIT.  The extraction stops after the loop.
            Exec (S, "COMMIT");
         exception
            when Sql_Error =>
               Rollback (S);
               raise;
         end Transaction;

         Found := True;
         Status := Success;
      end if;
   finally
      Free_Path (TS_Copy);
   end;
exception
   when Sql_Error =>
      Status := Db_Error;
end Reindex_Session;
