package Test_Types
  with SPARK_Mode
is

   subtype Small_Index is Integer range 1 .. 100;

   subtype My_Small is Integer range -100 .. 100;

   type Small_Matrix is
     array (Small_Index range <>, Small_Index range <>) of Integer;

end Test_Types;
