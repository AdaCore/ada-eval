pragma Assertion_Policy (Ignore);

package body C_Strings with SPARK_Mode is

   -----------------------------
   -- Is_Nul_Terminated_Ghost --
   -----------------------------

   function Is_Nul_Terminated_Ghost (Item : String) return Boolean is separate;

end C_Strings;
