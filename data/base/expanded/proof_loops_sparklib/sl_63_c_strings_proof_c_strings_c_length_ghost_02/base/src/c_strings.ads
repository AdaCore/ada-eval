pragma Assertion_Policy (Ignore);

with Interfaces.C;         use Interfaces.C;

package C_Strings with SPARK_Mode, Always_Terminates is

   function Is_Nul_Terminated_Ghost (Item : String) return Boolean with
     Ghost,
     Import,
     Global => null,
     Post => Is_Nul_Terminated_Ghost'Result =
         (for some C of Item => C = To_Ada (nul));
   --  Ghost function returns True if Item contains nul, and is False otherwise

   function C_Length_Ghost (Item : String) return Natural with
     Ghost,
     Pre  => Is_Nul_Terminated_Ghost (Item),
     Post => C_Length_Ghost'Result <= Item'Last - Item'First
       and then Item (Item'First + C_Length_Ghost'Result) = To_Ada (nul)
       and then (for all J in Item'First .. Item'First + C_Length_Ghost'Result
                   when J /= Item'First + C_Length_Ghost'Result =>
                     Item (J) /= To_Ada (nul));
   --  Ghost function to compute the length of a string up to the first nul
   --  character.

end C_Strings;
