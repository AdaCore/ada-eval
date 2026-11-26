pragma Assertion_Policy (Ignore);
separate (C_Strings)
function Is_Nul_Terminated_Ghost (Item : String) return Boolean is
begin
   for J in Item'Range loop
      if Item (J) = Character'First then
         return True;
      end if;
   end loop;

   return False;
end Is_Nul_Terminated_Ghost;
