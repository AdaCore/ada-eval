pragma Assertion_Policy (Ignore);

with Interfaces.C;         use Interfaces.C;

package C_Strings with SPARK_Mode, Always_Terminates is

   function C_Length_Ghost (Item : char_array) return size_t with
     Ghost,
     Pre  => Is_Nul_Terminated (Item),
     Post => C_Length_Ghost'Result = Interfaces.C.C_Length_Ghost (Item),
     Annotate => (GNATprove, Inline_For_Proof);

end C_Strings;
