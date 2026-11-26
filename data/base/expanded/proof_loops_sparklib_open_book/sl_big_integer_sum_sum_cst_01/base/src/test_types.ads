package Test_Types
  with SPARK_Mode
is

   subtype Small_Index is Integer range 1 .. 100;

   subtype My_Small is Integer range -100 .. 100;

   type Small_Array is array (Small_Index range <>) of Integer;
   --  There cannot be any overflow in the computation if Result_In_Range does
   --  not overflow.

end Test_Types;
