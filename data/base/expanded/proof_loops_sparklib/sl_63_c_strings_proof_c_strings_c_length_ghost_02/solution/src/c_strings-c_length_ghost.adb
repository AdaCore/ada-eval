pragma Assertion_Policy (Ignore);
separate (C_Strings)
function C_Length_Ghost (Item : String) return Natural is
begin
   for J in Item'Range loop
      if Item (J) = Character'First then
         return J - Item'First;
      end if;

      pragma Loop_Invariant
        (for all K in Item'First .. J => Item (K) /= Character'First);
   end loop;

   raise Program_Error;
end C_Length_Ghost;
