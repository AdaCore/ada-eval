with Ada.Assertions; use Ada.Assertions;

with Show_Permutation; use Show_Permutation;

procedure Tests is
   procedure Check_Assertions_Enabled is
   begin
      begin
         pragma Assert (False, "Should raise");
      exception
         when others =>
            return; -- properly raised
      end;
      raise Program_Error with "Assertions not enabled";
   end Check_Assertions_Enabled;

   procedure Test_Swap is
      Single_Item  : Permutation (1 .. 1) := [1];
      Longer_Array : Permutation := [1, 2, 3, 4, 5];
   begin
      Swap (Single_Item, 1, 1);
      pragma
        Assert (Single_Item (1) = 1, "Expected [1], got " & Single_Item'Image);
      Swap (Longer_Array, Longer_Array'First, Longer_Array'Last);
      pragma
        Assert
          (Longer_Array = [5, 2, 3, 4, 1],
           "Expected [5, 2, 3, 4, 1], got " & Longer_Array'Image);
   end Test_Swap;

   procedure Test_Init is
      Single_Item_1 : Permutation (1 .. 1);
      Single_Item_2 : Permutation (Positive'Last .. Positive'Last);
      Longer_Array  : Permutation (1 .. 5);
      Empty_Array   : Permutation (1 .. 0);
   begin
      Init (Single_Item_1);
      pragma
        Assert
          (Single_Item_1 = [1], "Expected [1], got " & Single_Item_1'Image);
      Init (Single_Item_2);
      pragma
        Assert
          (Single_Item_2 = [Positive'Last],
           "Expected ["
             & Positive'Last'Image
             & "], got "
             & Single_Item_2'Image);
      Init (Longer_Array);
      pragma
        Assert
          (Longer_Array = [1, 2, 3, 4, 5],
           "Expected [1, 2, 3, 4, 5], got " & Longer_Array'Image);
      Init (Empty_Array);
      pragma
        Assert
          (Empty_Array'Length = 0,
           "Expected empty array, got length " & Empty_Array'Length'Image);
   end Test_Init;

   procedure Test_Cyclic_Permutation is
   begin
      pragma
        Assert
          (Cyclic_Permutation (0) = [],
           "Expected [], got " & Cyclic_Permutation (0)'Image);
      pragma
        Assert
          (Cyclic_Permutation (1) = [1],
           "Expected [1], got " & Cyclic_Permutation (1)'Image);
      pragma
        Assert
          (Cyclic_Permutation (2) = [2, 1],
           "Expected [2, 1], got " & Cyclic_Permutation (2)'Image);
      pragma
        Assert
          (Cyclic_Permutation (3) = [2, 3, 1],
           "Expected [2, 3, 1], got " & Cyclic_Permutation (3)'Image);
      pragma
        Assert
          (Cyclic_Permutation (10) = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
           "Expected [2, 3, 4, 5, 6, 7, 8, 9, 10, 1], got "
             & Cyclic_Permutation (10)'Image);
   end Test_Cyclic_Permutation;

begin
   Check_Assertions_Enabled;
   Test_Swap;
   Test_Init;
   Test_Cyclic_Permutation;
end Tests;
