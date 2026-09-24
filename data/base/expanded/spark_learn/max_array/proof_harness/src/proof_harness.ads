with Array_Utils;

package Proof_Harness is

   function Check_Max_Array (A : Array_Utils.Array_Of_Naturals) return Natural
   with
     Pre  => A'Length > 0,
     Post => (for all I in A'Range => Check_Max_Array'Result >= A (I));

end Proof_Harness;
