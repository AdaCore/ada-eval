pragma Assertion_Policy (Ignore);
separate (C_Strings)
function C_Length_Ghost (Item : String) return Natural is
begin
   for J in Item'Range loop
      if Item (J) = Character'First then
         return J - Item'First;
      end if;
   end loop;

   raise Program_Error;
end C_Length_Ghost;
