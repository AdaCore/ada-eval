package body Proof_Harness is

   function Check_Max_Array (A : Array_Utils.Array_Of_Naturals) return Natural
   is
   begin
      return Array_Utils.Max_Array (A);
   end Check_Max_Array;

end Proof_Harness;
