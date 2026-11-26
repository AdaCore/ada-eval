pragma Assertion_Policy (Ignore);

with Interfaces.C;         use Interfaces.C;

package C_Strings with SPARK_Mode, Always_Terminates is

   function Is_Nul_Terminated_Ghost (Item : String) return Boolean with
     Ghost,
     Post => Is_Nul_Terminated_Ghost'Result =
         (for some C of Item => C = To_Ada (nul));
   --  Ghost function returns True if Item contains nul, and is False otherwise

end C_Strings;
