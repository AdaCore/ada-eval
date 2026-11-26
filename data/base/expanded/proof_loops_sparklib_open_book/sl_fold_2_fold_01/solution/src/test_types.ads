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

   Fst2 : constant Integer := Id (1);
   Lst2 : constant Integer := Id (200);

   subtype My_Index_2 is Integer range Fst2 .. Lst2;

   type My_Matrix is array (My_Index range <>, My_Index_2 range <>) of Integer;

end Test_Types;
