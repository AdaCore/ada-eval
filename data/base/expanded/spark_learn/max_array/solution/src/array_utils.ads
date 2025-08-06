package Array_Utils is

   type Array_Of_Naturals is array (Integer range <>) of Natural;

   function Max_Array (A : Array_Of_Naturals) return Natural
   with
     Pre  => A'Length > 0,
     Post => (for all I in A'Range => Max_Array'Result >= A (I));

end Array_Utils;
