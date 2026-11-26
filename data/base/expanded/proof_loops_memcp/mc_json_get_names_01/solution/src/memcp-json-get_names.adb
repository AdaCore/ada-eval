separate (Memcp.Json)
function Get_Names
  (M : access constant Types.JSON_Value) return Memcp.Store.Name_List
is
   use Memcp.Store;
   Result : Name_List := Name_Vectors.Empty_Vector;
begin
   if M = null or else Types.Kind (M) /= Types.Array_Kind then
      return Result;
   end if;
   for I in 1 .. Types.Length (M) loop
      pragma
        Loop_Invariant
          (Name_Vectors.Length (Result)
           <= Name_Vectors.Capacity_Range (I - 1));
      declare
         E : constant access constant Types.JSON_Value := Types.Get (M, I);
      begin
         if E /= null and then Types.Kind (E) = Types.String_Kind then
            declare
               S : constant String := Types.Value (E);
            begin
               Name_Vectors.Append (Result, (Len => S'Length, Value => S));
            end;
         end if;
      end;
   end loop;
   return Result;
end Get_Names;
