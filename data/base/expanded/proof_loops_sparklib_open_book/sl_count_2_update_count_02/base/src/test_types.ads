package Test_Types
  with SPARK_Mode
is

   subtype Small_Index is Integer range 1 .. 100;

   --  There cannot be any overflow in the computation if Result_In_Range does
   --  not overflow.

   type Small_Matrix is
     array (Small_Index range <>, Small_Index range <>) of Integer;

end Test_Types;
