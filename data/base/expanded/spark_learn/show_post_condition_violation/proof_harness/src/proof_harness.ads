package Proof_Harness is

   --  Allow the solution to exclude Integer'First using the pre-condition
   procedure Check_Absolute (X : in out Integer)
   with Pre => X /= Integer'First, Post => X >= 0;

end Proof_Harness;
