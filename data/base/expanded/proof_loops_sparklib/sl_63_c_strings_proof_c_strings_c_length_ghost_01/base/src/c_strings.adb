pragma Assertion_Policy (Ignore);

package body C_Strings with SPARK_Mode is

   --------------------
   -- C_Length_Ghost --
   --------------------

   function C_Length_Ghost (Item : char_array) return size_t is separate;

end C_Strings;
