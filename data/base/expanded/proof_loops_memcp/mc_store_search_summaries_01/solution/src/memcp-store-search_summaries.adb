separate (Memcp.Store)
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
is
   Len_P       : constant Name_Vectors.Capacity_Range :=
     Name_Vectors.Length (Projects);
   Has_Filters : constant Boolean :=
     Len_P > 0 or else Has_Since or else Has_Until;
   Lim         : constant Natural := Natural'Min (Limit, Max_Search_Limit);
   Over        : constant Natural := (if Has_Filters then Lim * 5 else Lim);
   Count       : Natural := 0;
begin
   Result := Summary_Hit_Vectors.Empty_Vector;

   if Lim = 0 or else Len_P > Max_Filter_Terms then
      Status := Success;
      return;
   end if;

   declare
      K1        : Sql.Statement;
      Have_Cand : Boolean;
   begin
      Prepare
        (S,
         "SELECT rowid, distance FROM summary_vec"
         & " WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
         K1);
      --  Upstream binds the packed query embedding to param 1 here.
      Bind (K1, 2, Row_Id (Over));

      --  One prepared per-row fetch, reset and rebound between candidates:
      --  the filtered over-fetch runs to Lim * 5 rows.
      declare
         M : Sql.Statement;
      begin
         Prepare (S, Fetch_Summary_SQL, M);

         loop
            pragma
              Loop_Invariant (Sql.Is_Valid (K1) and then Sql.Is_Valid (M));
            --  Both handles are written by a `finally`, as in
            --  Recent_Headers,
            --  and so are in this loop's frame.
            Step_Row (K1, Have_Cand);
            exit when not Have_Cand;
            exit when Count >= Lim;
            exit when
              Summary_Hit_Vectors.Length (Result)
              = Summary_Hit_Vectors.Last_Count;
            declare
               Rid       : constant Row_Id := Sql.Column_Int64 (K1, 0);
               Dist      : constant Interfaces.IEEE_Float_64 :=
                 Sql.Column_Double (K1, 1);
               Have_Meta : Boolean;
            begin
               Reset (M);
               Bind (M, 1, Rid);
               Step_Row (M, Have_Meta);
               if Have_Meta then
                  declare
                     Proj   : Sql.Text_Ptr := Sql.Column_Text (M, 1);
                     Sess   : Sql.Text_Ptr := Sql.Column_Text (M, 2);
                     Crea   : Sql.Text_Ptr := Sql.Column_Text (M, 3);
                     Head   : Sql.Text_Ptr := Sql.Column_Text (M, 4);
                     Bod    : Sql.Text_Ptr := Sql.Column_Text (M, 5);
                     Kind   : Sql.Text_Ptr := Sql.Column_Text (M, 6);
                     Null_S : constant Boolean := Sql.Column_Is_Null (M, 2);
                     Has_S  : constant Boolean := not Null_S;
                     --  Two steps, as in Fetch_Summary: a volatile column
                     --  read may not be an operand of `not`.
                     Passes : constant Boolean :=
                       (Len_P = 0 or else Contains (Projects, Proj.all))
                       and then (not Has_Since or else Crea.all >= Since)
                       and then
                         (not Has_Until or else Crea.all <= Until_At);
                  begin
                     if Passes then
                        Summary_Hit_Vectors.Append
                          (Result,
                           Summary_Hit'
                             (Project_Len => Proj.all'Length,
                              Session_Len => Sess.all'Length,
                              Created_Len => Crea.all'Length,
                              Header_Len  => Head.all'Length,
                              Body_Len    => Bod.all'Length,
                              Kind_Len    => Kind.all'Length,
                              Id          => Rid,
                              Has_Session => Has_S,
                              Project     => Proj.all,
                              Session     => Sess.all,
                              Created_At  => Crea.all,
                              Header      => Head.all,
                              Content     => Bod.all,
                              Kind        => Kind.all,
                              Distance    => Dist));
                        Count := Count + 1;
                     end if;
                     Sql.Free (Proj);
                     Sql.Free (Sess);
                     Sql.Free (Crea);
                     Sql.Free (Head);
                     Sql.Free (Bod);
                     Sql.Free (Kind);
                  end;
               end if;
            end;
         end loop;
      finally
         Sql.Finalize (M);
      end;

      --  Stopping early -- enough hits, or at capacity -- is as much a
      --  success as exhausting the candidates: a partial read is still a
      --  read. Only an unreadable row is a failure, and that raises.
      Status := Success;
   finally
      Sql.Finalize (K1);
   end;
exception
   when Sql_Error =>
      Status := Db_Error;
end Search_Summaries;
