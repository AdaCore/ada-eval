pragma Assertion_Policy (Ignore);
separate (C_Strings)
function C_Length_Ghost (Item : char_array) return size_t is
begin
   for J in Item'Range loop
      if Item (J) = nul then
         return J - Item'First;
      end if;
   end loop;

   raise Program_Error;
end C_Length_Ghost;
