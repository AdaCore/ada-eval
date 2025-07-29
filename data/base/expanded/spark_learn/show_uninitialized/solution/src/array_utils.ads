package Array_Utils is

   type Array_Of_Naturals is array (Integer range <>) of Natural;

   function Max_Array (A : Array_Of_Naturals) return Natural;

end Array_Utils;
