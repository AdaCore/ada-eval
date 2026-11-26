separate (Memcp.Store)
procedure Search_Chunks
  (S           : Store;
   Projects    : Name_List;
   Session_Ids : Name_List;
   Limit       : Natural;
   Has_Since   : Boolean;
   Since       : String;
   Has_Until   : Boolean;
   Until_At    : String;
   Result      : out Chunk_Hit_List;
   Status      : out Op_Status)
is
   Len_P       : constant Name_Vectors.Capacity_Range :=
     Name_Vectors.Length (Projects);
   Len_S       : constant Name_Vectors.Capacity_Range :=
     Name_Vectors.Length (Session_Ids);
   Has_Filters : constant Boolean :=
     Len_P > 0 or else Len_S > 0 or else Has_Since or else Has_Until;
   Lim         : constant Natural := Natural'Min (Limit, Max_Search_Limit);
   Over        : constant Natural := (if Has_Filters then Lim * 5 else Lim);
   Count       : Natural := 0;
begin
   Result := Chunk_Hit_Vectors.Empty_Vector;

   if Lim = 0
     or else Len_P > Max_Filter_Terms
     or else Len_S > Max_Filter_Terms
   then
      Status := Success;
      return;
   end if;

   declare
      K1        : Sql.Statement;
      Have_Cand : Boolean;
   begin
      Prepare
        (S,
         "SELECT rowid, distance FROM chunk_vec"
         & " WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
         K1);
      --  Upstream binds the packed query embedding to param 1 here.
      Bind (K1, 2, Row_Id (Over));

      --  One prepared per-row fetch, reset and rebound between candidates,
      --  as in Search_Summaries.
      declare
         M : Sql.Statement;
      begin
         Prepare (S, Chunk_By_Id_SQL, M);

         loop
            Step_Row (K1, Have_Cand);
            exit when not Have_Cand;
            exit when Count >= Lim;
            exit when
              Chunk_Hit_Vectors.Length (Result)
              = Chunk_Hit_Vectors.Last_Count;
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
                     Sr_C   : constant Row_Id := Sql.Column_Int64 (M, 1);
                     Ord_C  : constant Row_Id := Sql.Column_Int64 (M, 3);
                     Proj   : Sql.Text_Ptr := Sql.Column_Text (M, 2);
                     Bod    : Sql.Text_Ptr := Sql.Column_Text (M, 4);
                     Crea   : Sql.Text_Ptr := Sql.Column_Text (M, 5);
                     Sess   : Sql.Text_Ptr := Sql.Column_Text (M, 6);
                     Passes : constant Boolean :=
                       (Len_P = 0 or else Contains (Projects, Proj.all))
                       and then
                         (Len_S = 0
                          or else Contains (Session_Ids, Sess.all))
                       and then (not Has_Since or else Crea.all >= Since)
                       and then
                         (not Has_Until or else Crea.all <= Until_At);
                  begin
                     if Passes then
                        Chunk_Hit_Vectors.Append
                          (Result,
                           Chunk_Hit'
                             (Project_Len    => Proj.all'Length,
                              Body_Len       => Bod.all'Length,
                              Created_Len    => Crea.all'Length,
                              Session_Len    => Sess.all'Length,
                              Id             => Rid,
                              Session_Row_Id => Sr_C,
                              Ordinal        => Ord_C,
                              Project        => Proj.all,
                              Content        => Bod.all,
                              Created_At     => Crea.all,
                              Session        => Sess.all,
                              Distance       => Dist));
                        Count := Count + 1;
                     end if;
                     Sql.Free (Proj);
                     Sql.Free (Bod);
                     Sql.Free (Crea);
                     Sql.Free (Sess);
                  end;
               end if;
            end;
         end loop;
      finally
         Sql.Finalize (M);
      end;

      --  As in Search_Summaries: stopping early is still a read.
      Status := Success;
   finally
      Sql.Finalize (K1);
   end;
exception
   when Sql_Error =>
      Status := Db_Error;
end Search_Chunks;
