pragma Assertion_Policy (Ignore);

package body C_Strings with SPARK_Mode is

   --------------------
   -- C_Length_Ghost --
   --------------------

   function C_Length_Ghost (Item : String) return Natural is separate;

end C_Strings;
