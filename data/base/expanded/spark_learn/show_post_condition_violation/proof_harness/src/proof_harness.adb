with Integer_Utils;

package body Proof_Harness is

   procedure Check_Absolute (X : in out Integer) is
   begin
      Integer_Utils.Absolute (X);
   end Check_Absolute;

end Proof_Harness;
