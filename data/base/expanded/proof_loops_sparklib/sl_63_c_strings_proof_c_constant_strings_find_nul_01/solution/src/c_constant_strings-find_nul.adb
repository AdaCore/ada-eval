pragma Assertion_Policy (Ignore);
separate (C_Constant_Strings)
function Find_Nul (Chars : char_array) return size_t is
begin
   for I in Chars'Range loop
      if Chars (I) = nul then
         return I;
      end if;
      pragma
        Loop_Invariant (for all J in Chars'First .. I => Chars (J) /= nul);
   end loop;
   raise Program_Error;
end Find_Nul;
