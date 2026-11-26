pragma Assertion_Policy (Ignore);

with Interfaces.C; use Interfaces.C;

package C_Constant_Strings
  with SPARK_Mode, Always_Terminates
is

   function Find_Nul (Chars : char_array) return size_t
   with
     Pre  => Is_Nul_Terminated (Chars),
     Post => Find_Nul'Result = Chars'First + C_Length_Ghost (Chars);

end C_Constant_Strings;
