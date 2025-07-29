package Search_Array is

   type Array_Of_Positives is array (Natural range <>) of Positive;

   Not_Found : exception;

   procedure Search_Array
     (A : Array_Of_Positives; E : Positive; Result : out Integer);

end Search_Array;
