package Test_Types with SPARK_Mode is

   package Nested is
      function Id (X : Integer) return Integer;
   private
      pragma SPARK_Mode (Off);
      function Id (X : Integer) return Integer is (X);
   end Nested;
   use Nested;

   Fst : constant Integer := Id (0);
   Lst : constant Integer := Id (100);

   subtype My_Index is Integer range Fst .. Lst;

   type My_Array is array (My_Index range <>) of Integer;

end Test_Types;
