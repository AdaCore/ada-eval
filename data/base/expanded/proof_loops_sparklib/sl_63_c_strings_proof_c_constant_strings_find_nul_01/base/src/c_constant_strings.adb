pragma Assertion_Policy (Ignore);

package body C_Constant_Strings
  with SPARK_Mode => On
is

   --------------
   -- Find_Nul --
   --------------

   function Find_Nul (Chars : char_array) return size_t is separate;

end C_Constant_Strings;
